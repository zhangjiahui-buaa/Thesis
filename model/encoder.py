from torchvision import models
import torch.nn as nn
import torch
import timm
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    MMBTConfig,
    MMBTForClassification,
    get_linear_schedule_with_warmup,
)
import torchvision
POOLING_BREAKDOWN = {1: (1, 1), 2: (2, 1), 3: (3, 1), 4: (2, 2), 5: (5, 1), 6: (3, 2), 7: (7, 1), 8: (4, 2), 9: (3, 3)}

#  TODO: 1. image feature extractor
#  2. transformer encoder

class CNN_Image_Encoder(nn.Module):
    def __init__(self):
        super(CNN_Image_Encoder, self).__init__()
        self.cnn = timm.create_model('resnet50', pretrained=True, num_classes=0)
        for para in self.cnn.parameter():
            assert para.require_grad is True

    def forward(self, x):
        return self.cnn(x)  # torch.Size([bz, 2048])


class Transformer_Image_Encoder(nn.Module):
    def __init__(self):
        super(Transformer_Image_Encoder, self).__init__()
        self.encoder = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=3)
        for para in self.encoder.parameter():
            assert para.require_grad is True

    def forward(self, x):
        return self.encoder.forward_features(x)  # torch.Size([bz, 768])


class ViT_Image_Encoder(nn.Module):
    def __init__(self):
        super(ViT_Image_Encoder, self).__init__()
        self.encoder = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=3)
        for para in self.encoder.parameter():
            assert para.require_grad is True

    def forward(self, x):
        return self.encoder.forward_features(x)  # torch.Size([bz, 768])

class MMBT_ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        model = torchvision.models.resnet152(pretrained=True)
        modules = list(model.children())[:-2]
        self.model = nn.Sequential(*modules)
        self.pool = nn.AdaptiveAvgPool2d(POOLING_BREAKDOWN[1]) ## output_size = (1,1)

    def forward(self, x):
        # Bx3x224x224 -> Bx2048x7x7 -> Bx2048xN -> BxNx2048
        out = self.pool(self.model(x))
        out = torch.flatten(out, start_dim=2)
        out = out.transpose(1, 2).contiguous()
        return out  # BxNx2048, N = 1

class MMBT(nn.Module):
    def __init__(self, num_labels):
        super(MMBT, self).__init__()
        self.image_encoder = MMBT_ImageEncoder()
        self.transformer_config = AutoConfig.from_pretrained("bert-base-uncased")
        self.transformer = AutoModel.from_pretrained(
            "bert-base-uncased", config=self.transformer_config
        )
        self.config = MMBTConfig(self.transformer_config, num_labels=num_labels)
        self.model = MMBTForClassification(self.config, self.transformer, self.image_encoder)


    def forward(self, input_image, input_text, label):
        # B x (text + image) -> B x 3
        return self.model(input_image,input_text, labels = label)


