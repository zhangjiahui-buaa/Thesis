from preprocess.MVSASingle.dataset import load_MVSA_data_iterator
from transformers import BertTokenizer
import torchvision
from tqdm import tqdm
from model.encoder import *
from model.decoder import *
from model.model import *
import torch.nn as nn
import torch
import torch.optim as optim
import logging
import time
import os


# TODO: re-construct the code structure
def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-task', '--task', help='text, image or multi', type=str, default="image")
    parser.add_argument('-multi_type', '--multi_type', help='if multi, encode image and text separately or together?',
                        type=str, default="separate")
    parser.add_argument('-dataset', '--dataset', help='mvsa or hateful', type=str, default="mvsa")
    parser.add_argument('-data_dir', '--data_dir', help='mvsa or hateful', type=str, default="datasets/MVSA_Single")
    parser.add_argument('-image_enc', '--image_enc', help='cnn, tranformer or vit', type=str, default="cnn")
    parser.add_argument('-text_enc', '--text_enc', help='lstm or bert', type=str, default="bert")
    parser.add_argument('-mixed_enc', '--mixed_enc', help='mmbt or unit', type=str, default="mmbt")
    parser.add_argument('-bert_version', '--bert_version', help='bert version', type=str, default="bert-base-uncased")
    parser.add_argument('-device', '--device', help='cpu or gpu', type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument('-enc_lr', '--enc_lr', help='learning rate of encoder', type=float, default=1e-3)
    parser.add_argument('-dec_lr', '--dec_lr', help='learning rate of decoder', type=float, default=1e-2)
    parser.add_argument('-epoch', '--epoch', help='training epochs', type=int, default=10)
    parser.add_argument('-log', '--log', help='path to save logging info', type=str, default="output/logging.log")
    parser.add_argument('-log_step', '--log_step', help='print logging info by step', type=int, default=10)
    parser.add_argument('-save_dir', '--save_dir', help='save directory', type=str, default="output/dir")
    args = parser.parse_args()
    return args


def get_train_and_dev_loader(args):
    if args.dataset == "mvsa":
        return load_MVSA_data_iterator('datasets/MVSA_Single',
                                       BertTokenizer.from_pretrained("bert-base-uncased"),
                                       torchvision.transforms.Compose(
                                           [torchvision.transforms.Resize(256),
                                            torchvision.transforms.CenterCrop(224),
                                            # torchvision.transforms.RandomHorizontalFlip(),
                                            torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize(
                                                [0.485, 0.456, 0.406],
                                                [0.229, 0.224, 0.225])
                                            ]),
                                       32,
                                       args.device,
                                       True,
                                       512)
    else:
        raise NotImplementedError("Dataset {} has not been implemented".format(args.dataset))


def evaluate(model: nn.Module, dev_loader, args, logger: logging.Logger):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for batch in dev_loader:
            output = model(**batch)

            loss, logits = output[:2]
            label = batch["combined_label"]
            pred = torch.argmax(logits, dim=-1)
            total += len(label)
            correct += (pred == label).sum()

    logger.info("Evaluate Result--Dev set Loss:{}, Dev set Accuract:{}".format(loss, correct / total))

    return correct / total


def train(model: nn.Module, train_loader, dev_loader, optimizer: optim.Optimizer, args, logger: logging.Logger) -> None:
    logger.info("**********Begin Training**********")
    best_accuracy = 0
    for epoch in args.epoch:
        logger.info('Begin Epoch: {}'.format(epoch))
        # train model
        model.train()
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            assert isinstance(batch, dict)
            output = model(**batch)
            loss = output[0]
            loss.backward()
            optimizer.step()
            if step % args.log_step == 0:
                logger.info("Epoch:{}, Step:{}, Training Loss:{}".format(epoch, step, loss))

        # evaluate and save best model
        accuracy = evaluate(model, dev_loader, args, logger)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            save_model(model, best_accuracy, logger, args)

    logger.info("**********Finish Training**********")
    logger.info("Best accuracy on dev set is {}".format(best_accuracy))


def get_model(args) -> nn.Module:
    if args.task == "text":
        return Text_Model(args).to(args.device)
    elif args.task == "image":
        return Image_Model(args).to(args.device)
    elif args.task == "multi":
        return MultiModal_Model(args).to(args.device)
    else:
        raise NotImplementedError("Unknown task type, only support text, image, multi")


def save_model(model: nn.Module, accuracy: float, logger: logging.Logger, args):
    logger.info("Saving model....")
    saved_path = os.path.join(args.save_dir, str(accuracy), ".pt")
    torch.save(model.state_dict(), saved_path)


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
    args.save_dir = os.path.join("output", str(cur_time))
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    logger = get_logger(args)
    logger.info("Begin Logging")

    logger.info("Loading training data and test data")
    train_data_loader, dev_data_loader = get_train_and_dev_loader(args=args)

    logger.info("Loading model")
    model = get_model(args=args)

    optimizer = optim.Adam(model.parameters(), 0.001)
    train(model, train_data_loader, optimizer, args, logger)


if __name__ == '__main__':
    main()
