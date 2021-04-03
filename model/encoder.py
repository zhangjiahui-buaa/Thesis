from torchvision import models
import torch.nn as nn
#  TODO: 1. image feature extractor
#  2. transformer encoder

class CNN_Image_Encoder(nn.Module):
    def __init__(self):
        super(CNN_Image_Encoder, self).__init__()
        self.cnn = models.resnet50(pretrained=True)

    def forward(self, x):
        return self.cnn(x)


class Transformer_Image_Encoder(nn.Module):
    def __init__(self):
        pass


