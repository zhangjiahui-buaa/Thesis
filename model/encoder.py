import torch.nn as nn
import torch
import timm
from transformers import (
    AutoConfig,
    AutoModel,
    MMBTConfig,
    MMBTForClassification,
    BertModel
)
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import torchvision
from model.utils import *

POOLING_BREAKDOWN = {1: (1, 1), 2: (2, 1), 3: (3, 1), 4: (2, 2), 5: (5, 1), 6: (3, 2), 7: (7, 1), 8: (4, 2), 9: (3, 3)}


#  TODO: 1. image feature extractor
#  2. transformer encoder

class CNN_Image_Encoder(nn.Module):
    def __init__(self, args):
        super(CNN_Image_Encoder, self).__init__()
        self.cnn = timm.create_model('resnet152', pretrained=args.image_enc_pre_trained, num_classes=0)
        self.transform = torchvision.transforms.Compose(
            [torchvision.transforms.Resize(256),
             torchvision.transforms.CenterCrop(224),
             torchvision.transforms.ToTensor(),
             torchvision.transforms.Normalize(
                 [0.485, 0.456, 0.406],
                 [0.229, 0.224, 0.225])
             ]) if args.image_enc_pre_trained is True else torchvision.transforms.Compose(
            [torchvision.transforms.Resize(256),
             torchvision.transforms.CenterCrop(224),
             torchvision.transforms.ToTensor(),
             torchvision.transforms.Normalize(
                 [0.500, 0.500, 0.500],
                 [0.500, 0.500, 0.500])
             ])

        for para in self.cnn.parameters():
            assert para.requires_grad is True

    def forward(self, x):
        return self.cnn(x)  # torch.Size([bz, 2048])


class Transformer_Image_Encoder(nn.Module):
    def __init__(self, args):
        super(Transformer_Image_Encoder, self).__init__()
        self.encoder = timm.create_model('vit_base_patch16_224', pretrained=args.image_enc_pre_trained,
                                         num_classes=args.label_num)
        self.config = resolve_data_config({}, model=self.encoder)
        self.transform = create_transform(**self.config)
        for para in self.encoder.parameters():
            assert para.requires_grad is True

    def forward(self, x):
        return self.encoder.forward_features(x)  # torch.Size([bz, 768])


class ViT_Image_Encoder(nn.Module):
    def __init__(self, args):
        super(ViT_Image_Encoder, self).__init__()
        self.encoder = timm.create_model('vit_base_patch16_224', pretrained=args.image_enc_pre_trained,
                                         num_classes=args.label_num)
        self.config = resolve_data_config({}, model=self.encoder)
        self.transform = create_transform(**self.config)
        for para in self.encoder.parameters():
            assert para.requires_grad is True

    def forward(self, x):
        return self.encoder.forward_features(x)  # torch.Size([bz, 768])


class SwinT_Image_Encoder(nn.Module):
    def __init__(self, args):
        super(SwinT_Image_Encoder, self).__init__()
        self.encoder = timm.create_model('swin_base_patch4_window7_224', pretrained=args.image_enc_pre_trained,
                                         num_classes=3)
        self.config = resolve_data_config({}, model=self.encoder)
        self.transform = create_transform(**self.config)
        for para in self.encoder.parameters():
            assert para.requires_grad is True

    def forward(self, x):
        return self.encoder.forward_features(x)  # torch.Size([bz, 1024])


class TNT_Image_Encoder(nn.Module):
    def __init__(self, args):
        super(TNT_Image_Encoder, self).__init__()
        self.encoder = timm.create_model('tnt_s_patch16_224', pretrained=args.image_enc_pre_trained,
                                         num_classes=args.label_num)
        self.config = resolve_data_config({}, model=self.encoder)
        self.transform = create_transform(**self.config)
        for para in self.encoder.parameters():
            assert para.requires_grad is True

    def forward(self, x):
        return self.encoder.forward_features(x)  # torch.Size([bz, 384])


class PiT_Image_Encoder(nn.Module):
    def __init__(self, args):
        super(PiT_Image_Encoder, self).__init__()
        self.encoder = timm.create_model('pit_b_224', pretrained=args.image_enc_pre_trained, num_classes=args.label_num)
        self.config = resolve_data_config({}, model=self.encoder)
        self.transform = create_transform(**self.config)
        for para in self.encoder.parameters():
            assert para.requires_grad is True

    def forward(self, x):
        return self.encoder.forward_features(x).squeeze(1)  # torch.Size([bz, 1024])


class MMBT_ImageEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        '''model = torchvision.models.resnet152(pretrained=True)
        modules = list(model.children())[:-2]
        self.model = nn.Sequential(*modules)
        self.pool = nn.AdaptiveAvgPool2d(POOLING_BREAKDOWN[1])  # output_size = (1,1)
        self.transform = torchvision.transforms.Compose(
            [torchvision.transforms.Resize(256),
             torchvision.transforms.CenterCrop(224),
             torchvision.transforms.ToTensor(),
             torchvision.transforms.Normalize(
                 [0.485, 0.456, 0.406],
                 [0.229, 0.224, 0.225])
             ])'''
        self.model, self.transform = choose_image_encoder(args)

    def forward(self, x):
        return self.model(x).unsqueeze(1)
        ''' Bx3x224x224 -> Bx2048x7x7 -> Bx2048xN -> BxNx2048
        out = self.pool(self.model(x))
        out = torch.flatten(out, start_dim=2)
        out = out.transpose(1, 2).contiguous()
        print(out.shape)
        return out   BxNx2048, N = 1'''


class MMBT(nn.Module):
    def __init__(self, args, num_labels):
        super(MMBT, self).__init__()
        self.image_encoder = MMBT_ImageEncoder(args)
        self.transformer_config = AutoConfig.from_pretrained("bert-base-uncased")
        self.transformer = AutoModel.from_pretrained(
            "bert-base-uncased", config=self.transformer_config
        )
        self.config = MMBTConfig(self.transformer_config, num_labels=num_labels, modal_hidden_size=Image_Feature_Size[args.image_enc])
        self.model = MMBTForClassification(self.config, self.transformer, self.image_encoder)

    def forward(self, input_image, input_text, labels):
        # B x (text + image) -> loss : B x 1 , logits: B x labels
        return self.model(input_image, input_text, labels=labels)


class BERT_Text_Encoder(nn.Module):
    def __init__(self, bert_version: str = "bert-base-uncased"):
        super(BERT_Text_Encoder, self).__init__()
        self.bert = BertModel.from_pretrained(bert_version)
        for para in self.bert.parameters():
            assert para.requires_grad is True

    def forward(self, **inputs):
        bert_outputs, _ = self.bert(
            inputs['input_token_ids'],
            attention_mask=inputs['input_token_ids'].ne(0))  # bz x len_seq x hidden_size

        cls_outputs = bert_outputs[:, 0, :]  # bz x hidden_size
        return cls_outputs


def choose_image_encoder(args):
    if args.image_enc == "cnn":
        model = CNN_Image_Encoder(args=args)
    elif args.image_enc == "transformer":
        model = Transformer_Image_Encoder(args=args)
    elif args.image_enc == "vit":
        model = ViT_Image_Encoder(args=args)
    elif args.image_enc == "swint":
        model = SwinT_Image_Encoder(args=args)
    elif args.image_enc == "tnt":
        model = TNT_Image_Encoder(args=args)
    elif args.image_enc == "pit":
        model = PiT_Image_Encoder(args=args)
    else:
        raise ValueError("unknown image encoder")
    return model, model.transform


def choose_multi_encoder(args):
    if args.mixed_enc == "mmbt":
        model = MMBT(args, args.label_num)
    else:
        raise ValueError("unknown multimodal encoder")
    return model, model.image_encoder.transform


def choose_text_encoder(args):
    if args.text_enc == "bert":
        return BERT_Text_Encoder()
    else:
        raise ValueError("unknown text encoder")
