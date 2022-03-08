import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
import wandb
#from torchvision import transforms as T
from tqdm import tqdm
import numpy as np
from dataset import pytorch_dataset
import pandas as pd
from models import *
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


# parser:

parser = argparse.ArgumentParser(description='Testing arguments')

parser.add_argument('-m', '--model',
                    metavar='model', help='which model to use in training: resnet50, vit-large, vit-base')

parser.add_argument('--device', default='cuda',
                    metavar='device', help='device used during testing')

parser.add_argument('--test_dir',
                    metavar='testing-directory', help='Directory of the testing csv')

parser.add_argument('--id', type=int,
                    metavar='id', help='id of the test')

parser.add_argument('--weights_dir',
                    metavar='weights_dir', help='Directory of weights')

parser.add_argument('--name', metavar='name',
                    help='Experiment name that logs into wandb')

parser.add_argument('--project_name',
                    metavar='project_name', help='Project name, utilized for logging purposes in W&B.')

parser.add_argument('--group',
                    metavar='group', help='Grouping argument for W&B init.')

parser.add_argument('-d', '--dataset_dir', required=True,
                    metavar='dataset_dir', help='Directory where the datasets are stored.')


args = parser.parse_args()


def testing(model, dataloader, criterion):

    print('Testing')

    model.eval()

    running_loss = 0.0

    y_true = []
    y_pred = []

    correct = 0

    with torch.no_grad():

        for data in tqdm(dataloader):

            x, y = data[0].to(args.device), data[1].to(args.device)
            y = y.unsqueeze(1)

            outputs = model(x)
            loss = criterion(outputs, y)

            running_loss += loss.item()

            outputs = model.sigmoid(outputs)

            outputs = torch.round(outputs)
            correct_ = (outputs == y).sum().item()
            correct += correct_

            y = y.squeeze(0)
            outputs = outputs.squeeze(0)

            y = y.cpu().item()
            outputs = outputs.cpu().item()

            y_true.append(y)
            y_pred.append(outputs)

    acc = 100. * correct/len(dataloader.dataset)
    print(f'test acc: {acc}')

    wandb.log({'test-loss': loss})

    print(f"Test loss is {loss}")

    return loss, y_true, y_pred


def log_metrics(y_true, y_pred):

    test_acc = accuracy_score(y_true, y_pred)
    test_f1 = f1_score(y_true, y_pred)
    test_prec = precision_score(y_true, y_pred)
    test_rec = recall_score(y_true, y_pred)
    test_auc = roc_auc_score(y_true, y_pred)

    wandb.log({
        'Accuracy': test_acc,
        'F1': test_f1,
        'Precision': test_prec,
        'Recall': test_rec,
        'ROC-AUC score': test_auc})


def to_numpy(y_true, y_pred):

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_true = y_true.astype(int)
    y_pred = np.rint(y_pred)

    return y_true, y_pred


def log_conf_matrix(y_true, y_pred):

    conf_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred)
    conf_matrix = pd.DataFrame(data=conf_matrix, columns=['A', 'B'])
    cf_matrix = wandb.Table(dataframe=conf_matrix)
    wandb.log({'conf_mat': cf_matrix})

# main def:


def main():

    # initialize w&b

    wandb.init(project=args.project_name, name=args.name,
               config=vars(args), group=args.group)

    # initialize model:

    if args.model == 'resnet50':
        model = resnet50()
    elif args.model == 'vit-large':
        model = vit_large()
    elif args.model == 'vit-base':
        model = vit_base()
    elif args.model == 'swin':
        model = swin_small()
    elif args.model == 'resnet101':
        model = resnet101()
    elif args.model == 'vit-small':
        model = vit_small()
    elif args.model == 'swin-tiny':
        model = swin_tiny()
    elif args.model == 'vit-tiny':
        model = vit_tiny()
    elif args.model == 'inception-v4':
        model = inception_v4()
    elif args.model == 'xception':
        model = xception()

    # load weights:

    model.load_state_dict(torch.load(args.weights_dir))

    # set the device:

    device = torch.device(
        args.device if torch.cuda.is_available() else 'cpu')

    model.to(device)

    # defining transforms:

    transforms = A.Compose([
        A.augmentations.geometric.resize.Resize(256, 256),
        A.augmentations.crops.transforms.CenterCrop(224, 224),
        A.Normalize(),
        ToTensorV2(),
    ])

    # define test dataset:

    test_dataset = pytorch_dataset.dataset2(
        args.dataset_dir, args.test_dir, transforms)

    # define data loaders:

    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # set the criterion:

    criterion = nn.BCEWithLogitsLoss()

    # testing

    test_loss, y_true, y_pred = testing(
        model=model, dataloader=test_dataloader, criterion=criterion)

    # converting lists into numpy arrays

    y_true, y_pred = to_numpy(y_true=y_true, y_pred=y_pred)

    # calculating and logging results:

    log_metrics(y_true=y_true, y_pred=y_pred)

    log_conf_matrix(y_true=y_true, y_pred=y_pred)

    print(f'Finished Testing with test loss = {test_loss}')


if __name__ == '__main__':
    main()
