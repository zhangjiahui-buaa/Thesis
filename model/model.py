from model.encoder import *
from model.decoder import *
import torch.nn as nn
from model.utils import *


class MultiModal_Model(nn.Module):
    def __init__(self, args):
        super(MultiModal_Model, self).__init__()
        self.args = args
        self.image_encoder = None
        self.text_encoder = None
        self.mixed_encoder = None
        self.decoder = None
        if args.multi_type == "separate":
            self.image_encoder = choose_image_encoder(args)
            self.text_encoder = choose_text_encoder(args)
            self.decoder = Image_and_Text_Decoder(Image_Feature_Size[args.image_encoder],
                                                  Text_Feature_Size[args.text_encoder], args.num_label)
        elif args.multi_type == "together":
            self.mixed_encoder = choose_multi_encoder(args)
        else:
            raise ValueError("unsupported multi type")

    def forward(self, **inputs):
        if self.image_encoder is not None:
            # inputs -> bz x num_labels
            return self.decoder(self.image_encoder(inputs["input_image"]), self.text_encoder(**inputs))
        else:
            return self.mixed_encoder(inputs["input_image"], inputs["input_text"], )


class Image_Model(nn.Module):
    def __init__(self, args):
        super(Image_Model, self).__init__()
        self.encoder = choose_image_encoder(args)
        self.decoder = Image_Decoder(Image_Feature_Size[args.image_encoder], self.args.label_num)

    def forward(self, **inputs):
        # bz x 3 x 224 x 224 -> bz x args.label_num
        return self.decoder(self.encoder(inputs["input_image"]))


class Text_Model(nn.Module):
    def __init__(self, args):
        super(Text_Model, self).__init__()
        self.encoder = BERT_Text_Encoder()
        self.decoder = Text_Decoder(label_num=args.label_num)

    def forward(self, **inputs):
        # inputs -> bz x args.label_num
        return self.decoder(self.encoder(**inputs))
