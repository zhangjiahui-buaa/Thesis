import torch.nn as nn
from typing import *


# TODO: feature fusion module
#   classification module

class CNN_Image_Decoder(nn.Module):
    def __init__(self):
        super(CNN_Image_Decoder, self).__init__()
        self.linear = nn.Linear(1000, 3)

    def forward(self, x):
        return self.linear(x)

