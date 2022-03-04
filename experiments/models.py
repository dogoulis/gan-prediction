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
        self.model.fc = nn.Linear(self.model.fc.in_features, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.model(x)
        return x

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.fc.parameters():
            param.requires_grad = True
    
    def unfreeze(self):
        for param in self.model.parameters():
            param.requires_grad = True

#ResNet101:
class resnet101(nn.Module):

    def __init__(self):
        super(resnet101, self).__init__()
        self.model = timm.create_model('resnet101', pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.model(x)
        return x

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.fc.parameters():
            param.requires_grad = True
    
    def unfreeze(self):
        for param in self.model.parameters():
            param.requires_grad = True

#ViT-large

class vit_large(nn.Module):

    def __init__(self):
        super(vit_large, self).__init__()
        self.model = timm.create_model('vit_large_patch16_224', pretrained=True)
        self.model.head = nn.Linear(self.model.head.in_features, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.model(x)
        return x

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.head.parameters():
            param.requires_grad = True
    
    def unfreeze(self):
        for param in self.model.parameters():
            param.requires_grad = True

#ViT-base

class vit_base(nn.Module):

    def __init__(self):
        super(vit_base, self).__init__()
        self.model = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.model.head = nn.Linear(self.model.head.in_features, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.model(x)
        return x

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.head.parameters():
            param.requires_grad = True
    
    def unfreeze(self):
        for param in self.model.parameters():
            param.requires_grad = True


#vit-small
class vit_small(nn.Module):

    def __init__(self):
        super(vit_small, self).__init__()
        self.model = timm.create_model('vit_small_patch16_224', pretrained=True)
        self.model.head = nn.Linear(self.model.head.in_features, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.model(x)
        return x

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.head.parameters():
            param.requires_grad = True
    
    def unfreeze(self):
        for param in self.model.parameters():
            param.requires_grad = True


#vit-tiny
class vit_tiny(nn.Module):

    def __init__(self):
        super(vit_tiny, self).__init__()
        self.model = timm.create_model('vit_tiny_patch16_224', pretrained=True)
        self.model.head = nn.Linear(self.model.head.in_features, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.model(x)
        return x

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.head.parameters():
            param.requires_grad = True
    
    def unfreeze(self):
        for param in self.model.parameters():
            param.requires_grad = True

#Swin-base-Transformer

class swin_small(nn.Module):

    def __init__(self):
        super(swin_small, self).__init__()
        self.model = timm.create_model('swin_small_patch4_window7_224', pretrained=True)
        self.model.head = nn.Linear(self.model.head.in_features, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.model(x)
        return x

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.head.parameters():
            param.requires_grad = True
    
    def unfreeze(self):
        for param in self.model.parameters():
            param.requires_grad = True


# swin tiny
class swin_tiny(nn.Module):

    def __init__(self):
        super(swin_tiny, self).__init__()
        self.model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True)
        self.model.head = nn.Linear(self.model.head.in_features, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.model(x)
        return x

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.head.parameters():
            param.requires_grad = True
    
    def unfreeze(self):
        for param in self.model.parameters():
            param.requires_grad = True


# inception

class inception_v4(nn.Module):

    def __init__(self):
        super(inception_v4, self).__init__()
        self.model = timm.create_model('inception_v4', pretrained=True)
        self.model.last_linear = nn.Linear(self.model.last_linear.in_features, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.model(x)
        return x

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.fc.parameters():
            param.requires_grad = True
    
    def unfreeze(self):
        for param in self.model.parameters():
            param.requires_grad = True