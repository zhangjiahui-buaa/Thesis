import torch.nn as nn
import torch
from typing import *


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
    def __init__(self, image_feature_size: int = 2048, text_feature_size: int = 768, label_num: int = 3):
        super(Image_and_Text_Decoder, self).__init__()
        self.linear = nn.Linear(image_feature_size + text_feature_size, label_num)
        for para in self.parameters():
            assert para.requires_grad is True

    def forward(self, image_tensor, text_tensor):
        return self.linear(torch.cat([image_tensor, text_tensor]))
