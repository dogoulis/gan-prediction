import os
import sys 
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
import wandb
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
 

sys.path.append('/home/dogoulis/WORKSPACE/datasets')
from pytorch_dataset import dataset2


# experiment configuration

CONFIG = {

    'batch_size':32,
    'Optimizer':'adam',
    'Loss':'BCE with Logits',
    'learning_rate': 1e-3,
    'weight_decay': 1e-5,
    'Model-Type': 'resnet',
    'GIQA': True,
    'device':'cuda'

}


# parser:
    
parser = argparse.ArgumentParser(description='Training arguments')

parser.add_argument('-b', '--batch_size', type=int, default=CONFIG['batch_size'],
                metavar='batch_size', help='input batch size for training (default: 32)')

parser.add_argument('-lr', '--learning_rate', type=float,default=CONFIG['learning_rate'],
                metavar='Learning Rate', help='learning rate of the optimizer')

parser.add_argument('-wd', '--weight_decay', type=float,default=CONFIG['weight_decay'],
                    metavar='Weight Decay', help='Weight decay of the optimizer')

parser.add_argument('-gq', '--giqa', type=bool, default=CONFIG['GIQA'],
                metavar='GIQA', help='Train with Giqa')

parser.add_argument('-d', '--device', default=CONFIG['device'],
                metavar='device', help='device used during training')


args = parser.parse_args()
CONFIG.update(vars(args))
 
# make the class for the model

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

# define training logic

def train_epoch(model, train_dataloader, CONFIG, optimizer, criterion):

    print('Training')
    model.train()
    running_loss = 0.0

    batch = 0

    for data in tqdm(train_dataloader):
        
        x, y = data[0].to(CONFIG['device']), data[1].to(CONFIG['device'])

        batch += 1 

        y = y.unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(x)


        loss = criterion(outputs,y)

        if batch%10 == 0:
            wandb.log({'train-step-loss':loss})

        loss.backward()        
        optimizer.step()

        running_loss += loss.item()

        
    train_loss = running_loss/len(train_dataloader.dataset)

    wandb.log({'train-epoch-loss':train_loss})

    return train_loss, data, outputs

# define validation logic

def validate_epoch(model, val_dataloader, CONFIG, criterion):
    print('Validating')

    model.eval()

    running_loss = 0.0

    batch = 0
    
    with torch.no_grad():

        for data in tqdm(val_dataloader):

            x, y = data[0].to(CONFIG['device']), data[1].to(CONFIG['device'])
            y = y.unsqueeze(1)

            batch += 1

             
            outputs = model(x)
            loss = criterion(outputs, y)
            if batch%10 == 0:
                wandb.log({'valid-step-loss':loss})
            
            running_loss += loss.item()

        
        val_loss = running_loss/len(val_dataloader.dataset)
        wandb.log({'valdiation-epoch-loss':val_loss})

        return val_loss, data, outputs

# MAIN def

def main():

    # initialize weights and biases:

    wandb.init(project='test-test', config=CONFIG)

    # initialize model:

    model = resnet50()

    # set the device

    device = torch.device(CONFIG['device'] if torch.cuda.is_available() else 'cpu')

    model.to(device)

    # defining transforms

    transforms = [A.Compose([
        # resize
            A.augmentations.geometric.resize.Resize(256, 256),
        # center crop:
            A.augmentations.crops.transforms.CenterCrop (224,224),
            A.Normalize(),
            ToTensorV2()]), 
            
            A.Compose([A.augmentations.geometric.resize.Resize(256, 256),
            A.augmentations.crops.transforms.CenterCrop (224,224),
            A.Normalize(),
            ToTensorV2()
    ])]
    
    # set the paths for training
        
    dataset_train = '/home/dogoulis/WORKSPACE/datasets/CSV/train_10k.csv'
    dataset_val =  '/home/dogoulis/WORKSPACE/datasets/CSV/validation_dataset.csv'

    train_dataset = dataset2(dataset_train ,transforms[0])

    val_dataset = dataset2(dataset_val, transforms[1])

    # defining data loaders:

    train_dataloader = DataLoader(train_dataset, batch_size=24, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=24, shuffle=True)

    # setting the model:

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)

    criterion = nn.BCEWithLogitsLoss()

    # directory:

    save_dir = './'
    
    
    n_epochs = 1


    for epoch in range(n_epochs):

        train_epoch_loss, _, _ = train_epoch(model, train_dataloader=train_dataloader,CONFIG=CONFIG,
                                 optimizer=optimizer, criterion=criterion)
        val_epoch_loss, _, _ = validate_epoch(model, val_dataloader=val_dataloader, CONFIG=CONFIG,
                                 criterion=criterion)

        torch.save(model.cpu().state_dict(), os.path.join(save_dir, 'epoch-{}.pt'.format(epoch)))

        print(train_epoch_loss, val_epoch_loss)

if __name__ == '__main__':
    main()