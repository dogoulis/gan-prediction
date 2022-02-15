import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

"""
This module contains the models that are used for experimentation.
Models are currently used from timm.
"""


#ResNet50:
class resnet50(nn.Module):

    def __init__(self):
        super(resnet50, self).__init__()
        self.model = timm.create_model('resnet50', pretrained=True)
        self.model.classification = nn.Linear(self.model.fc.out_features, 1)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x = self.model(x)
        x = self.dropout(x)
        x = self.model.classification(x)
        return x

#ViT-large

class vit_large(nn.Module):

    def __init__(self):
        super(vit_large, self).__init__()
        self.model = timm.create_model('vit_large_patch16_224', pretrained=True)
        self.model.classification = nn.Linear(self.model.head.out_features, 1)
        self.dropout = nn.Dropout(p=0.3)
    
    def forward(self, x):
        x = self.model(x)
        x = self.dropout(x)
        x = self.model.classification(x)
        return x

#ViT-base

class vit_base(nn.Module):

    def __init__(self):
        super(vit_base, self).__init__()
        self.model = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.model.classification = nn.Linear(self.model.head.out_features, 1)
        self.dropout = nn.Dropout(p=0.3)
    
    def forward(self, x):
        x = self.model(x)
        x = self.dropout(x)
        x = self.model.classification(x)
        return x

#Swin-base-Transformer

class swin_base(nn.Module):

    def __init__(self):
        super(swin_base, self).__init__()
        self.model = timm.create_model('swin_base_patch4_window7_224')
        self.model.classification = nn.Linear(self.model.head.out_features, 1)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x = self.model(x)
        x = self.dropout(x)
        x = self.model.classification(x)
        return x