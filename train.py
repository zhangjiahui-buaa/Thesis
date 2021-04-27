from preprocess.dataset import load_MVSA_data_iterator, load_Hate_data_iterator
from transformers import BertTokenizer
from model.model import *
import torch.nn as nn
import torch
import torch.optim as optim
from transformers import AdamW
import logging
import time
import os
from torchmetrics import AUROC


# TODO: re-construct the code structure
def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-task', '--task', help='text, image or multi', type=str, default="multi")
    parser.add_argument('-multi_type', '--multi_type', help='if multi, encode image and text separately or together?',
                        type=str, default="separate")
    parser.add_argument('-dataset', '--dataset', help='mvsa or hateful', type=str, default="hateful")
    parser.add_argument('-label_num', '--label_num', help='number of label', type=int, default=2)
    parser.add_argument('-image_enc', '--image_enc', help='cnn, tranformer or vit', type=str, default="cnn")
    parser.add_argument('-image_enc_pre_trained', '--image_enc_pre_trained', help='true or false', type=bool,
                        default=True)
    parser.add_argument('-text_enc', '--text_enc', help='lstm or bert', type=str, default="bert")
    parser.add_argument('-mixed_enc', '--mixed_enc', help='mmbt or unit', type=str, default="mmbt")
    parser.add_argument('-model_checkpoint', '--model_checkpoint', help='path to model', type=str, default=None)
    parser.add_argument('-bert_version', '--bert_version', help='bert version', type=str, default="bert-base-uncased")
    parser.add_argument('-batch_size', '--batch_size', help='batch size', type=int, default="32")
    parser.add_argument('-weight_decay', '--weight_decay', help='weight decay', type=float, default=0.01)
    parser.add_argument('-label_smoothing', '--label_smoothing', help='label smoothing', type=float, default=0.1)
    parser.add_argument('-device', '--device', help='cpu or gpu', type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument('-enc_lr', '--enc_lr', help='learning rate of encoder', type=float, default=3e-5)
    parser.add_argument('-dec_lr', '--dec_lr', help='learning rate of decoder', type=float, default=1e-3)
    parser.add_argument('-epoch', '--epoch', help='training epochs', type=int, default=20)
    parser.add_argument('-log', '--log', help='path to save logging info', type=str, default="output/logging.log")
    parser.add_argument('-log_step', '--log_step', help='print logging info by step', type=int, default=50)
    parser.add_argument('-save_dir', '--save_dir', help='save directory', type=str, default="output/dir")
    args = parser.parse_args()
    return args


def get_train_and_dev_loader(args, transform):
    if args.dataset == "mvsa":
        args.label_num = 3
        return load_MVSA_data_iterator('datasets/MVSA_Single',
                                       BertTokenizer.from_pretrained(args.bert_version),
                                       transform,
                                       args.batch_size,
                                       args.device,
                                       True,
                                       512)
    elif args.dataset == "hateful":
        args.label_num = 2
        return load_Hate_data_iterator('datasets/Hateful',
                                       BertTokenizer.from_pretrained(
                                           args.bert_version),
                                       transform,
                                       args.batch_size,
                                       args.device,
                                       True,
                                       512)
    else:
        raise NotImplementedError("Dataset {} has not been implemented".format(args.dataset))


def infer(model: nn.Module, test_loader, args, logger: logging.Logger):  # infer on test set
    model.eval()
    pass


def evaluate(model: nn.Module, dev_loader, args, logger: logging.Logger):
    model.eval()
    total = 0
    total_loss = 0
    correct = 0
    all_prob = torch.tensor([[0, 0]]).to(args.device)
    all_label = torch.tensor([0]).to(args.device)
    auroc = AUROC(num_classes=2, pos_label=1)
    with torch.no_grad():
        for batch in dev_loader:
            loss, pred, logits = model.compute_loss(**batch)
            prob = F.softmax(logits, dim=-1)
            all_prob = torch.cat([all_prob, prob])
            if args.task == 'multi':
                label = batch["combined_label"]
            elif args.task == 'image':
                label = batch['image_label']
            else:
                label = batch['text_label']
            all_label = torch.cat([all_label, label])
            total += len(label)
            total_loss += loss * len(label)
            correct += (pred == label).sum().item()
    all_label = all_label[1:]
    all_prob = all_prob[1:]
    auroc_res = auroc(all_prob, all_label)
    logger.info(
        "Evaluate Result--Dev set Loss:{:.4f}, Dev set Accuracy:{:.4f}, Dev set AUROC:{:.4f}".format(
            total_loss / total, correct / total, auroc_res))

    return correct / total, total_loss / total, auroc_res


def train(model: nn.Module, train_loader, dev_loader, optimizers: List[optim.Optimizer], args,
          logger: logging.Logger) -> None:
    logger.info("**********Begin Training**********")
    corresponding_accuracy = 0
    best_auroc = 0
    for epoch in range(args.epoch):
        logger.info('Begin Epoch: {}'.format(epoch))
        # train model
        model.train()
        for step, batch in enumerate(train_loader):
            for optimizer in optimizers:
                optimizer.zero_grad()
            assert isinstance(batch, dict)
            loss = model.compute_loss(**batch)[0]
            loss.backward()
            for optimizer in optimizers:
                optimizer.step()
            if step % args.log_step == 0:
                logger.info("Epoch:{}, Step:{}, Training Loss:{:.4f}".format(epoch, step, loss))

        # evaluate and save best model
        accuracy, _, auroc = evaluate(model, dev_loader, args, logger)
        if auroc > best_auroc:
            best_auroc = auroc
            corresponding_accuracy = accuracy
            save_model(model, corresponding_accuracy, logger, args)

    logger.info("**********Finish Training**********")
    logger.info("Best auroc on dev set is {:.4f}".format(best_auroc))
    logger.info("Corresponding accuracy on dev set is {:.4f}".format(corresponding_accuracy))


def get_model_transform_and_optimizer(args, logger: logging.Logger):
    image_transform = None
    optimizers = []
    if args.task == "text":
        model = Text_Model(args).to(args.device)
        optimizers.append(AdamW(model.encoder.parameters(), args.enc_lr, weight_decay=args.weight_decay))
        optimizers.append(AdamW(model.decoder.parameters(), args.dec_lr, weight_decay=args.weight_decay))
    elif args.task == "image":
        model = Image_Model(args).to(args.device)
        image_transform = model.image_transform
        optimizers.append(AdamW(model.encoder.parameters(), args.enc_lr, weight_decay=args.weight_decay))
        optimizers.append(AdamW(model.decoder.parameters(), args.dec_lr, weight_decay=args.weight_decay))
    elif args.task == "multi":
        model = MultiModal_Model(args).to(args.device)
        image_transform = model.image_transform
        if args.multi_type == "separate":
            optimizers.append(AdamW(model.image_encoder.parameters(), args.enc_lr, weight_decay=args.weight_decay))
            optimizers.append(AdamW(model.text_encoder.parameters(), args.enc_lr, weight_decay=args.weight_decay))
            optimizers.append(AdamW(model.decoder.parameters(), args.dec_lr, weight_decay=args.weight_decay))
        else:
            optimizers.append(
                AdamW(model.mixed_encoder.model.mmbt.parameters(), args.enc_lr, weight_decay=args.weight_decay))
            optimizers.append(
                AdamW(model.mixed_encoder.model.classifier.parameters(), args.dec_lr, weight_decay=args.weight_decay))
    else:
        raise NotImplementedError("Unknown task type, only support text, image, multi")
    if args.model_checkpoint is not None:
        model.load_state_dict(torch.load(args.model_checkpoint, map_location=args.device))
        logger.info('Initialize {} from checkpoint {} over.'.format(model, args.model_checkpoint))
    else:
        logger.info('Initialize {} randomly.'.format(model))
    return model, image_transform, optimizers


def save_model(model: nn.Module, auroc: float, logger: logging.Logger, args):
    saved_path = os.path.join(args.save_dir, "model.pt")
    logger.info("Saving model at {}".format(saved_path))
    torch.save(model.state_dict(), saved_path)
    logger.info("Done!")


def get_logger(args):
    logger = logging.getLogger()
    fh = logging.FileHandler(os.path.join(args.save_dir, "Training.log"))
    ch = logging.StreamHandler()
    logger.addHandler(fh)
    logger.addHandler(ch)
    log_format = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
    fh.setFormatter(log_format)
    ch.setFormatter(log_format)
    logger.setLevel(logging.INFO)
    return logger


def main():
    args = parse_args()

    cur_time = int(time.time())
    args.save_dir = os.path.join("output",
                                 "{}_{}_{}_{}_{}".format(str(cur_time), args.task, args.multi_type, args.image_enc,
                                                         args.text_enc))
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    logger = get_logger(args)

    logger.info("Begin Logging")
    for k in list(vars(args).keys()):
        logger.info('%s: %s' % (k, vars(args)[k]))

    logger.info("Loading model, optimizers and image tranformation")
    model, transform, optimizers = get_model_transform_and_optimizer(args=args, logger=logger)

    logger.info("Loading data")
    data_loaders = get_train_and_dev_loader(args=args, transform=transform)

    train(model, data_loaders[0], data_loaders[1], optimizers, args, logger)

    # infer(model, data_loader[2], args, logger)


if __name__ == '__main__':
    main()
