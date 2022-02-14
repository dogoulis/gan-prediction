import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
import wandb
import timm
from torchvision import transforms as T
from tqdm import tqdm
import numpy as np
from dataset import pytorch_dataset
import pandas as pd

# testing configuration

CONFIG = {

    'Model-Type': 'resnet',
    'GIQA': True,
    'device': 'cuda'
}

# parser:

parser = argparse.ArgumentParser(description='Testing arguments')

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

# make the class for the model


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

    wandb.init(project='TESTING', config=CONFIG)

    # initialize model:

    model = resnet50()

    # load weights:

    model.load_state_dict(torch.load(args.weights_dir))

    # set the device:

    device = torch.device(
        CONFIG['device'] if torch.cuda.is_available() else 'cpu')

    model.to(device)

    # defining transforms:

    normalization = T.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
    transforms = T.Compose([
                    T.ToTensor(),
                    normalization
                ])
    

    # define test dir:

    dataset_test = args.test_dir

    # define test dataset:

    test_dataset = pytorch_dataset.dataset2(dataset_test, transforms)

    # define data loaders:

    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    # set the criterion:

    criterion = nn.BCELoss()

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
