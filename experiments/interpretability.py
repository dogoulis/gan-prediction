import captum
import torch
import torch.nn as nn
import argparse
import timm
from PIL import Image
from torchvision import transforms
from captum.attr import IntegratedGradients
from captum.attr import GradientShap
from captum.attr import Occlusion
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz
from matplotlib.colors import LinearSegmentedColormap
import numpy as np


# parser:

parser = argparse.ArgumentParser(description='Testing arguments')

parser.add_argument('--weights_dir',
                    metavar='weights_dir', help='Directory of weights')

parser.add_argument('--img', 
                    metavar='img_dir', help='Path of the image')


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

    # define transforms

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    # take a random image:

    img_dir = args.img

    img = Image.open(img_dir)

    transformed_img = transform(img)

    input = transformed_img.unsqueeze(0)

    # pass the image to the model:

    output = model(input)

    # integrated gradients model:

    integrated_gradients = IntegratedGradients(model)

    attributions_ig = integrated_gradients.attribute(input)

    default_cmap = LinearSegmentedColormap.from_list('custom blue',
                                                     [(0, '#ffffff'),
                                                      (0.25, '#000000'),
                                                         (1, '#000000')], N=256)

    _ = viz.visualize_image_attr(np.transpose(attributions_ig.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                                 np.transpose(transformed_img.squeeze(
                                 ).cpu().detach().numpy(), (1, 2, 0)),
                                 method='heat_map',
                                 cmap=default_cmap,
                                 show_colorbar=True,
                                 sign='positive',
                                 outlier_perc=1)


if __name__ == '__main__':
    main()
