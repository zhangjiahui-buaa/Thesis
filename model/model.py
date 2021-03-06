from model.encoder import *
from model.decoder import *
import torch.nn as nn
from model.utils import *
import torch.nn.functional as F
import torch


class LabelSmoothingLoss(nn.Module):  # label_smoothing
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


class MultiModal_Model(nn.Module):
    def __init__(self, args):
        super(MultiModal_Model, self).__init__()
        self.args = args
        self.image_encoder = None
        self.text_encoder = None
        self.mixed_encoder = None
        self.decoder = None
        if args.multi_type == "separate":
            self.image_encoder, self.image_transform = choose_image_encoder(args)
            self.text_encoder = choose_text_encoder(args)
            self.decoder = Image_and_Text_Decoder(args, Image_Feature_Size[args.image_enc],
                                                  Text_Feature_Size[args.text_enc], args.label_num)
        elif args.multi_type == "together":
            self.mixed_encoder, self.image_transform = choose_multi_encoder(args)
        else:
            raise ValueError("unsupported multi type")

    def forward(self, **inputs):
        if self.image_encoder is not None:
            # inputs -> bz x num_labels(logits)
            return self.decoder(self.image_encoder(inputs["input_image"]), self.text_encoder(**inputs))
        else:
            loss, logits = self.mixed_encoder(inputs["input_image"], inputs["input_token_ids"],
                                              inputs['combined_label'])
            return logits

    def compute_loss(self, **inputs):
        logits = self.forward(**inputs)
        pred = torch.argmax(logits, dim=-1)
        loss_fn = LabelSmoothingLoss(self.args.label_num, self.args.label_smoothing)
        loss = loss_fn(logits, inputs['combined_label'])
        return loss, pred, logits

    def predict(self, **inputs):
        logits = self.forward(**inputs)
        pred = torch.argmax(logits, dim=-1)
        prob = F.softmax(logits)
        return pred, prob


class Image_Model(nn.Module):
    def __init__(self, args):
        super(Image_Model, self).__init__()
        self.args = args
        self.encoder, self.image_transform = choose_image_encoder(args)
        self.decoder = Image_Decoder(Image_Feature_Size[args.image_enc], args.label_num)

    def forward(self, **inputs):
        # bz x 3 x 224 x 224 -> bz x args.label_num
        return self.decoder(self.encoder(inputs["input_image"]))

    def compute_loss(self, **inputs):
        logits = self.forward(**inputs)
        pred = torch.argmax(logits, dim=-1)
        loss_fn = LabelSmoothingLoss(self.args.label_num, self.args.label_smoothing)
        loss = loss_fn(logits, inputs['image_label'])
        return loss, pred, logits

    def predict(self, **inputs):
        logits = self.forward(**inputs)
        pred = torch.argmax(logits, dim=-1)
        return pred, logits


class Text_Model(nn.Module):
    def __init__(self, args):
        super(Text_Model, self).__init__()
        self.args = args
        self.encoder = BERT_Text_Encoder()
        self.decoder = Text_Decoder(label_num=args.label_num)

    def forward(self, **inputs):
        # inputs -> bz x args.label_num
        return self.decoder(self.encoder(**inputs))

    def compute_loss(self, **inputs):
        logits = self.forward(**inputs)
        pred = torch.argmax(logits, dim=-1)
        loss_fn = LabelSmoothingLoss(self.args.label_num, self.args.label_smoothing)
        loss = loss_fn(logits, inputs['text_label'])
        return loss, pred, logits

    def predict(self, **inputs):
        logits = self.forward(**inputs)
        pred = torch.argmax(logits, dim=-1)
        return pred, logits
