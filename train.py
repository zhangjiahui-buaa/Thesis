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


# TODO: re-construct the code structure
def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-task', '--task', help='text, image or multi', type=str, default="image")
    parser.add_argument('-multi_type', '--multi_type', help='if multi, encode image and text separately or together?',
                        type=str, default="separate")
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


def get_train_and_dev_loader(args):
    pass


def evaluate():
    pass


def train(model: nn.Module, train_loader, optimizer: optim.Optimizer, args, logger: logging.Logger) -> None:
    logger.info("**********Begin Training**********")
    for epoch in args.epoch:
        logger.info('Begin Epoch: {}'.format(epoch))
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            assert isinstance(batch, dict)
            output = model(**batch)
            loss = output[0]
            loss.backward()
            optimizer.step()
            if step % args.log_step == 0:
                logger.info("Epoch:{}, Step:{}, Training Loss:{}".format(epoch, step, loss))
        evaluate()
        save_model()

    pass


def get_model(args) -> nn.Module:
    if args.task == "text":
        return Text_Model(args)
    elif args.task == "image":
        return Image_Model(args)
    elif args.task == "multi":
        return MultiModal_Model(args)
    else:
        raise NotImplementedError("Unknown task type, only support text, image, multi")


def save_model():
    pass


def get_logger():
    logger = logging.getLogger()
    fh = logging.FileHandler("output/output.log")
    ch = logging.StreamHandler()
    logger.addHandler(fh)
    logger.addHandler(ch)
    log_format = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
    fh.setFormatter(log_format)
    ch.setFormatter(log_format)
    logger.setLevel(logging.INFO)
    return logger


def main():
    logger = get_logger()
    logger.info("Begin Logging")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_data_loader, dev_data_loader = load_MVSA_data_iterator('datasets/MVSA_Single',
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
                                                                 device,
                                                                 True,
                                                                 512)

    # image_encoder = ViT_Image_Encoder().to(device)
    # image_decoder = Image_Decoder(768).to(device)
    classifier = MMBT(3).to(device)
    # encoder_optimizer = optim.Adam(image_encoder.parameters(), 0.00003)
    # decoder_optimizer = optim.Adam(image_decoder.parameters(), 0.001)
    classifier_optimizer = optim.Adam(classifier.parameters(), 0.001)
    for epoch in range(10):
        # image_encoder.train()
        # image_decoder.train()
        classifier.train()
        for step, batch in enumerate(train_data_loader):
            # encoder_optimizer.zero_grad()
            # decoder_optimizer.zero_grad()
            classifier_optimizer.zero_grad()
            image_input = batch['input_image']
            text_input = batch['input_token_ids']
            label = batch['combined_label'].squeeze(1)
            outputs = classifier.forward(image_input, text_input, labels=label)
            loss, logits = outputs[:2]
            '''image_output = image_decoder(image_encoder(image_input))
            prob = F.softmax(image_output, dim=-1)
            gold = batch['image_label'].squeeze(1)
            loss = F.cross_entropy(prob, gold)'''
            loss.backward()

            # encoder_optimizer.step()
            # decoder_optimizer.step()
            classifier_optimizer.step()
            if step % 10 == 0:
                print(loss)
        with torch.no_grad():
            # image_encoder.eval()
            # image_decoder.eval()
            classifier.eval()
            total = 0
            correct = 0
            for batch in tqdm(dev_data_loader):
                image_input = batch['input_image']
                text_input = batch['input_token_ids']
                label = batch['combined_label'].squeeze(1)
                outputs = classifier(image_input, text_input, labels=label)
                _, logits = outputs[:2]
                #  image_output = image_decoder(image_encoder(image_input))
                # pred = torch.argmax(image_output, dim=-1)
                # gold = batch['image_label'].squeeze(1)
                pred = torch.argmax(logits, dim=-1)
                total += len(image_input)
                correct += (pred == label).sum()
            print("accuracy {}".format(correct / total))


if __name__ == '__main__':
    main()
