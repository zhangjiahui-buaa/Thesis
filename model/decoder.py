import torch.nn as nn
from typing import *


# TODO: feature fusion module
#   classification module

class CNN_Image_Decoder(nn.Module):
    def __init__(self, feature_size: int):
        super(CNN_Image_Decoder, self).__init__()
        self.linear = nn.Linear(feature_size, 3)
        for para in self.parameters():
            assert para.requires_grad is True
    def forward(self, x):
        return self.linear(x)

