import captum
import torch
import torch.nn as nn
import argparse
import timm


# parser:

parser = argparse.ArgumentParser(description='Testing arguments')

parser.add_argument('--weights_dir',
                    metavar='weights_dir', help='Directory of weights')


args = parser.parse_args()

# model:


class resnet50(nn.Module):

    def __init__(self):
        super(resnet50, self).__init__()
        self.model = timm.create_model('resnet50', pretrained=False)
        self.model.classification = nn.Linear(self.model.fc.out_features, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x = self.model(x)
        x = self.dropout(x)
        x = self.model.classification(x)
        x = self.sigmoid(x)
        return x


def main():

    # load model:

    model = resnet50()

    # load weights:

    model.load_state_dict(torch.load(args.weights_dir))
    
    # GradCam Method:

    grad_cam = captum.attr.GuidedGradCam(model, layer=model._modules['model'].layer4[-1].conv3)

    #set an input:

    img = torch.randn(2, 3, 32, 32, requires_grad=True)

    # calculate the attribution:

    attribution = grad_cam.attribute(input, 1)

    # print

    print(attribution)
    


if __name__ == '__main__':
    main()
