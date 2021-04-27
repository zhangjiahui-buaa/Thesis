import torch.nn as nn
import torch
from typing import *
import math
import torch.nn.functional as F
from model.utils import *


# TODO: feature fusion module
#   classification module

class Image_Decoder(nn.Module):
    def __init__(self, feature_size: int = 2048, label_num: int = 3):
        super(Image_Decoder, self).__init__()
        self.linear = nn.Linear(feature_size, label_num)
        for para in self.parameters():
            assert para.requires_grad is True

    def forward(self, x):
        return self.linear(x)


class Text_Decoder(nn.Module):
    def __init__(self, feature_size: int = 768, label_num: int = 3):
        super(Text_Decoder, self).__init__()
        self.linear = nn.Linear(feature_size, label_num)
        for para in self.parameters():
            assert para.requires_grad is True

    def forward(self, x):
        return self.linear(x)


class Image_and_Text_Decoder(nn.Module):  # TODO: weighted sum feature fusion
    def __init__(self, args, image_feature_size: int = 2048, text_feature_size: int = 768, label_num: int = 3):
        super(Image_and_Text_Decoder, self).__init__()
        self.args = args
        self.linear_dc_classifier = None
        self.linear_image = None
        self.linear_text = None
        self.linear_classifier = None
        self.attention = None
        self.gelu = nn.GELU()
        if args.dec_mode == "DC":
            self.linear_dc_classifier = nn.Linear(image_feature_size + text_feature_size, label_num)
        else:
            assert Image_Feature_Hidden_Size == Text_Feature_Hidden_Size
            self.linear_image = nn.Linear(image_feature_size, Image_Feature_Hidden_Size)
            self.linear_text = nn.Linear(text_feature_size, Text_Feature_Hidden_Size)
            self.linear_classifier = nn.Linear(Image_Feature_Hidden_Size + Text_Feature_Hidden_Size, label_num)
            if args.dec_mode == "STC":
                self.attention = nn.MultiheadAttention(Image_Feature_Hidden_Size, num_heads=2, dropout=args.dropout)

        for para in self.parameters():
            assert para.requires_grad is True

    def forward(self, image_tensor, text_tensor):
        # input size: bz x image_feature_size, bz x text_feature_size
        # return size: bz x label_num
        if self.args.dec_mode == "DC":
            return self.linear_dc_classifier(torch.cat([image_tensor, text_tensor], dim=-1))
        else:
            image_tensor_embed = self.gelu(self.linear_image(image_tensor))
            text_tensor_embed = self.gelu(self.linear_text(text_tensor))
            if self.args.dec_mode == "STC":  # apply self-attention
                attention_input = torch.cat([image_tensor_embed.unsqueeze(0), text_tensor_embed.unsqueeze(0)])
                attn_output, _ = self.attention(attention_input, attention_input, attention_input)
                image_tensor_embed, text_tensor_embed = attn_output[0].squeeze(0), attn_output[1].squeeze(0)
            return self.linear_classifier(torch.cat([image_tensor_embed, text_tensor_embed], dim=-1))
