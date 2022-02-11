import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
import wandb
import timm
from tqdm import tqdm
from dataset import pytorch_dataset
from torchvision import transforms as T
from models import resnet50, vit_base, vit_large

# experiment configuration

CONFIG = {

    'batch_size':32,
    'Optimizer':'adam',
    'Loss':'BCE with Logits',
    'learning_rate': 1e-3,
    'weight_decay': 1e-5,
    'Model-Type': 'resnet',
    'GIQA': 'yes',
    'device':'cuda'

}


# parser:
    
parser = argparse.ArgumentParser(description='Training arguments')

parser.add_argument('-m', '--model',
                metavar='model', help='which model to use in training: resnet50, vit-large, vit-base')

parser.add_argument('-b', '--batch_size', type=int, default=CONFIG['batch_size'],
                metavar='batch_size', help='input batch size for training (default: 32)')

parser.add_argument('-lr', '--learning_rate', type=float,default=CONFIG['learning_rate'],
                metavar='Learning Rate', help='learning rate of the optimizer')

parser.add_argument('-wd', '--weight_decay', type=float,default=CONFIG['weight_decay'],
                    metavar='Weight Decay', help='Weight decay of the optimizer')

parser.add_argument('-gq', '--giqa', default=CONFIG['GIQA'],
                metavar='GIQA', help='Train with Giqa')

parser.add_argument('-d', '--device', default=CONFIG['device'],
                metavar='device', help='device used during training')

parser.add_argument('--train_dir', metavar='train-dir', help='training dataset path for csv')

parser.add_argument('--valid_dir', metavar='valid-dir', help='validation dataset path for csv')

parser.add_argument('--save_dir', metavar='save-dir', help='save directory path')

args = parser.parse_args()
CONFIG.update(vars(args))

# define training logic

def train_epoch(model, train_dataloader, CONFIG, optimizer, criterion):

    print('Training')

    model.to(CONFIG['device'])
    model.train()
    running_loss = 0.0

    batch = 0

    for data in tqdm(train_dataloader):
        
        x, y = data[0].to(CONFIG['device']), data[1].to(CONFIG['device'])

        batch += 1 

        y = y.unsqueeze(1)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
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

    model.to(CONFIG['device'])
    model.eval()

    running_loss = 0.0

    correct = 0
    
    with torch.no_grad():

        for data in tqdm(val_dataloader):

            x, y = data[0].to(CONFIG['device']), data[1].to(CONFIG['device'])
            y = y.unsqueeze(1)
             
            outputs = model(x)
            loss = criterion(outputs, y)
            #loss calculation over batch
            running_loss += loss.item()
            #accuracy calcilation over batch
            outputs = torch.round(outputs)
            correct_ = (outputs == y).sum().item()
            correct += correct_

        val_loss = running_loss/len(val_dataloader.dataset)
        wandb.log({'valdiation-epoch-loss':val_loss})
        acc = 100. * correct/len(val_dataloader.dataset)
        wandb.log({'validation-accuracy':acc})

        return val_loss, data, outputs

# MAIN def

def main():

    # initialize weights and biases:

    wandb.init(project='torch-run-.61-10k', config=CONFIG)

    # initialize model:

    if args.model=='resnet50':
        model = resnet50()
    elif args.model=='vit-large':
        model = vit_large()
    elif args.model=='vit-base':
        model = vit_base()

    # defining transforms
    '''
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
    '''

    normalization = T.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
    transforms = T.Compose(
                [
                    T.RandomResizedCrop(224, scale=(0.7, 1.0)),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    normalization,
                ]
            )
    
    # set the paths for training
        
    
    dataset_train = args.train_dir
    
    dataset_val = args.valid_dir

    train_dataset = pytorch_dataset.dataset2(dataset_train ,transforms)

    val_dataset = pytorch_dataset.dataset2(dataset_val, transforms)

    # defining data loaders:

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

    # setting the model:

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    criterion = nn.BCEWithLogitsLoss()

    # directory:

    save_dir = args.save_dir
    
    
    n_epochs = 1


    for epoch in range(n_epochs):

        train_epoch_loss, _, _ = train_epoch(model, train_dataloader=train_dataloader,CONFIG=CONFIG,
                                 optimizer=optimizer, criterion=criterion)
        val_epoch_loss, _, _ = validate_epoch(model, val_dataloader=val_dataloader, CONFIG=CONFIG,
                                 criterion=criterion)

        torch.save(model.cpu().state_dict(), os.path.join(save_dir, f'epoch-{epoch}.pt'))

        print(train_epoch_loss, val_epoch_loss)

if __name__ == '__main__':
    main()
