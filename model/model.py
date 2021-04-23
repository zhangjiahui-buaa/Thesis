from model.encoder import *
from model.decoder import *
import torch.nn as nn
from model.utils import *
import torch.nn.functional as F
import torch


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
            self.decoder = Image_and_Text_Decoder(Image_Feature_Size[args.image_enc],
                                                  Text_Feature_Size[args.text_enc], args.label_num)
        elif args.multi_type == "together":
            self.mixed_encoder = choose_multi_encoder(args)
        else:
            raise ValueError("unsupported multi type")

    def forward(self, **inputs):
        if self.image_encoder is not None:
            # inputs -> bz x num_labels(logits)
            return self.decoder(self.image_encoder(inputs["input_image"]), self.text_encoder(**inputs))
        else:
            loss, logits = self.mixed_encoder(inputs["input_image"], inputs["input_text"], inputs['combined_label'])
            return logits

    def compute_loss(self, **inputs):
        logits = self.forward(**inputs)
        pred = torch.argmax(logits, dim=-1)
        prob = F.softmax(logits, dim=-1)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(prob, inputs['combined_label'])
        return loss, pred

    def predict(self, **inputs):
        logits = self.forward(**inputs)
        pred = torch.argmax(logits, dim=-1)
        return pred


class Image_Model(nn.Module):
    def __init__(self, args):
        super(Image_Model, self).__init__()
        self.encoder, self.image_transform = choose_image_encoder(args)
        self.decoder = Image_Decoder(Image_Feature_Size[self.args.image_enc], self.args.label_num)

    def forward(self, **inputs):
        # bz x 3 x 224 x 224 -> bz x args.label_num
        return self.decoder(self.encoder(inputs["input_image"]))

    def compute_loss(self, **inputs):
        logits = self.forward(**inputs)
        pred = torch.argmax(logits, dim=-1)
        prob = F.softmax(logits, dim=-1)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(prob, inputs['image_label'])
        return loss, pred

    def predict(self, **inputs):
        logits = self.forward(**inputs)
        pred = torch.argmax(logits, dim=-1)
        return pred, logits


class Text_Model(nn.Module):
    def __init__(self, args):
        super(Text_Model, self).__init__()
        self.encoder = BERT_Text_Encoder()
        self.decoder = Text_Decoder(label_num=args.label_num)

    def forward(self, **inputs):
        # inputs -> bz x args.label_num
        return self.decoder(self.encoder(**inputs))

    def compute_loss(self, **inputs):
        logits = self.forward(**inputs)
        pred = torch.argmax(logits, dim=-1)
        prob = F.softmax(logits, dim=-1)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(prob, inputs['text_label'])
        return loss, pred

    def predict(self, **inputs):
        logits = self.forward(**inputs)
        pred = torch.argmax(logits, dim=-1)
        return pred, logits
