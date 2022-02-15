import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
import wandb
#from torchvision import transforms as T
from tqdm import tqdm
import numpy as np
from dataset import pytorch_dataset
import pandas as pd
from models import *
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

# testing configuration

CONFIG = {

    'Model-Type': 'resnet',
    'GIQA': True,
    'device': 'cuda'
}

# parser:

parser = argparse.ArgumentParser(description='Testing arguments')

parser.add_argument('-m', '--model',
                    metavar='model', help='which model to use in training: resnet50, vit-large, vit-base')

parser.add_argument('--giqa', type=bool, default=CONFIG['GIQA'],
                    metavar='GIQA', help='Train with Giqa')

parser.add_argument('--device', default=CONFIG['device'],
                    metavar='device', help='device used during testing')

parser.add_argument('--test_dir',
                    metavar='testing-directory', help='Directory of the testing csv')

parser.add_argument('--id', type=int,
                    metavar='id', help='id of the test')

parser.add_argument('--weights_dir',
                    metavar='weights_dir', help='Directory of weights')


args = parser.parse_args()
CONFIG.update(vars(args))


def testing(model, dataloader, criterion):

    print('Testing')

    model.eval()

    running_loss = 0.0

    y_true = []
    y_pred = []

    with torch.no_grad():

        for data in tqdm(dataloader):

            x, y = data[0].to(CONFIG['device']), data[1].to(CONFIG['device'])
            y = y.unsqueeze(1)

            with torch.cuda.amp.autocast():
                outputs = model(x)
                loss = criterion(outputs, y)

            running_loss += loss.item()

            y = y.squeeze(0)
            outputs = outputs.squeeze(0)

            y = y.cpu().item()
            outputs = outputs.cpu().item()

            y_true.append(y)
            y_pred.append(outputs)

    test_loss = running_loss/len(dataloader.dataset)

    wandb.log({'test-loss': test_loss})

    print(f"Test loss is {test_loss}")

    return test_loss, y_true, y_pred


def log_metrics(y_true, y_pred):

    test_acc = accuracy_score(y_true, y_pred)
    test_f1 = f1_score(y_true, y_pred)
    test_prec = precision_score(y_true, y_pred)
    test_rec = recall_score(y_true, y_pred)

    wandb.log({
        'Accuracy': test_acc,
        'F1': test_f1,
        'Precision': test_prec,
        'Recall': test_rec})


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

    wandb.init(project='first-test', config=CONFIG)

    # initialize model:

    if args.model == 'resnet50':
        model = resnet50()
    elif args.model == 'vit-large':
        model = vit_large()
    elif args.model == 'vit-base':
        model = vit_base()
    elif args.model == 'swin':
        model = swin_base()
    
    # load weights:

    model.load_state_dict(torch.load(args.weights_dir))

    # set the device:

    device = torch.device(
        CONFIG['device'] if torch.cuda.is_available() else 'cpu')

    model.to(device)

    # defining transforms:
    transforms = A.Compose([
        A.augmentations.geometric.resize.Resize(256, 256),
        A.augmentations.crops.transforms.CenterCrop(224, 224),
        A.augmentations.transforms.HorizontalFlip(),
        A.Normalize(),
        ToTensorV2(),
    ])

    # define test dir:

    dataset_test = args.test_dir

    # define test dataset:

    test_dataset = pytorch_dataset.dataset2(dataset_test, transforms)

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
