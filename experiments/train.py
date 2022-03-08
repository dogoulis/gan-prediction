import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
import wandb
from tqdm import tqdm
from dataset import pytorch_dataset
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from models import *

# experiment configuration

CONFIG = {

    'batch_size': 32,
    'Optimizer': 'adam',
    'Loss': 'BCE with Logits',
    'learning_rate': 1e-3,
    'weight_decay': 2e-5,
    'device': 'cuda'

}


# parser:

parser = argparse.ArgumentParser(description='Training arguments')

parser.add_argument('--project_name',
                    metavar='project_name', help='Project name, utilized for logging purposes in W&B.')

parser.add_argument('-m', '--model',
                    metavar='model', help='which model to use in training: resnet50, vit-large, vit-base, swin, vit-small, swin0tiny, vit-tiny, inception-v4')

parser.add_argument('-e', '--epochs', type=int,
                    metavar='epochs', help='Number of epochs')

parser.add_argument('-b', '--batch_size', type=int, default=CONFIG['batch_size'],
                    metavar='batch_size', help='input batch size for training (default: 32)')

parser.add_argument('-lr', '--learning_rate', type=float, default=CONFIG['learning_rate'],
                    metavar='Learning Rate', help='learning rate of the optimizer (default: 1e-3)')

parser.add_argument('-wd', '--weight_decay', type=float, default=CONFIG['weight_decay'],
                    metavar='Weight Decay', help='Weight decay of the optimizer (default: 2e-5)')

parser.add_argument('-gq', '--giqa',
                    metavar='GIQA', help='Train with Giqa, only for logging purposes')

parser.add_argument('-d', '--device', default=CONFIG['device'],
                    metavar='device', help='device used during training (default: "cuda")')

parser.add_argument('--train_dir', metavar='train-dir',
                    help='training dataset path for csv')

parser.add_argument('--valid_dir', metavar='valid-dir',
                    help='validation dataset path for csv')

parser.add_argument('--save_dir', metavar='save-dir',
                    help='save directory path')

parser.add_argument('--name', metavar='name',
                    help='Experiment name that logs into wandb')

parser.add_argument('--pretrained', metavar='pretrained',
                    help='Flag for adding an argument if the model loaded for training does not load as pretrained from timm and need to load the weights from directory.\
                        Currently works for swin-base model.')

parser.add_argument('--weights_dir', metavar='weight_dir',
                    help='Directory of the weights for the model. Currently works for swin-base model')

parser.add_argument('--iso')
args = parser.parse_args()
CONFIG.update(vars(args))

# define training logic


def train_epoch(model, train_dataloader, CONFIG, optimizer, criterion, scheduler):

    print('Training')

    model.to(CONFIG['device'])

    # to train only the classification layer:
    # model.freeze()

    model.train()

    running_loss = 0.0

    running_loss_per_10 = 0.0

    batch = 0

    fp16_scaler = torch.cuda.amp.GradScaler()

    for data in tqdm(train_dataloader):

        x, y = data[0].to(CONFIG['device']), data[1].to(CONFIG['device'])

        batch += 1

        y = y.unsqueeze(1)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            outputs = model(x)

            loss = criterion(outputs, y)

        fp16_scaler.scale(loss).backward()
        fp16_scaler.step(optimizer)
        fp16_scaler.update()

        running_loss += loss.item()

        running_loss_per_10 += loss.item()
        #change the position of the scheduler:
        #scheduler.step()

        # log mean loss for the last 10 batches:
        if batch % 10 == 0:
            wandb.log({'train-step-loss': running_loss_per_10/10.0})
            running_loss_per_10 = 0.0

    train_loss = running_loss/batch

    wandb.log({'train-epoch-loss': train_loss})

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

            # loss calculation over batch
            running_loss += loss.item()
            # accuracy calculation over batch
            outputs = model.sigmoid(outputs)
            outputs = torch.round(outputs)
            correct_ = (outputs == y).sum().item()
            correct += correct_

        val_loss = running_loss/len(val_dataloader.dataset)
        wandb.log({'valdiation-epoch-loss': val_loss})
        acc = 100. * correct/len(val_dataloader.dataset)
        wandb.log({'validation-accuracy': acc})

        return val_loss, data, outputs

# MAIN def


def main():

    # initialize weights and biases:

    wandb.init(project=args.project_name, config=CONFIG,
               name=args.name, save_code=True)

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
    elif args.model== 'vit-small':
        model = vit_small()
    elif args.model== 'swin-tiny':
        model = swin_tiny()
    elif args.model== 'vit-tiny':
        model = vit_tiny()
    elif args.model == 'inception-v4':
        model = inception_v4()
    elif args.model =='xception':
        model = xception()

    if args.pretrained:
        model.load_state_dict(torch.load(args.weights_dir))

    # add Wang augmentations pipeline transformed into albumentations:

    train_transforms = A.Compose([
            A.augmentations.geometric.resize.Resize(256,256),
            A.augmentations.transforms.GaussianBlur(sigma_limit=(0.0, 3.0), p=0.5),
            A.augmentations.transforms.ImageCompression(
                quality_lower=30, quality_upper=100, p=0.5),
            A.augmentations.crops.transforms.RandomCrop(224, 224),
            A.augmentations.transforms.HorizontalFlip(),
            A.Normalize(),
            ToTensorV2(),
        ])

    valid_transforms = A.Compose([
            A.augmentations.geometric.resize.Resize(256,256),
            A.augmentations.crops.transforms.CenterCrop(224, 224),
            A.Normalize(),
            ToTensorV2(),
        ])
    
    if args.iso:
                
        train_transforms = A.Compose([
            A.augmentations.geometric.resize.Resize(256,256),
            A.OneOf([
            A.augmentations.transforms.GaussianBlur(sigma_limit=(0.0, 3.0), p=0.5),
            A.augmentations.transforms.ImageCompression(
                quality_lower=30, quality_upper=100, p=0.5),
            A.augmentations.transforms.ISONoise(p=0.5)]),
            A.augmentations.crops.transforms.RandomCrop(224, 224),
            A.augmentations.transforms.HorizontalFlip(),
            A.Normalize(),
            ToTensorV2(),
        ])

        valid_transforms = A.Compose([
            A.augmentations.geometric.resize.Resize(256,256),
            A.augmentations.crops.transforms.CenterCrop(224, 224),
            A.Normalize(),
            ToTensorV2(),
        ])

    # set the paths for training

    dataset_train = args.train_dir

    dataset_val = args.valid_dir

    train_dataset = pytorch_dataset.dataset2(dataset_train, train_transforms)

    val_dataset = pytorch_dataset.dataset2(dataset_val, valid_transforms)

    # defining data loaders:

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False)

    # setting the optimizer:

    # optimizer = torch.optim.Adam(
    # filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate, weight_decay=args.weight_decay)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # setting the scheduler:

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer=optimizer, step_size=5, gamma=0.1)

    criterion = nn.BCEWithLogitsLoss()

    # directory:

    save_dir = args.save_dir
    print(save_dir)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    else:
        pass

    n_epochs = args.epochs

    # set value for min-loss:
    min_loss = float('inf')

    for epoch in range(n_epochs):

        wandb.log({'epoch': epoch})

        train_epoch_loss, _, _ = train_epoch(model, train_dataloader=train_dataloader, CONFIG=CONFIG,
                                             optimizer=optimizer, criterion=criterion, scheduler=scheduler)
        val_epoch_loss, _, _ = validate_epoch(model, val_dataloader=val_dataloader, CONFIG=CONFIG,
                                              criterion=criterion)

        if val_epoch_loss < min_loss:
            min_loss = val_epoch_loss
            torch.save(model.cpu().state_dict(), os.path.join(
                save_dir, 'best-ckpt.pt'))

        print(f'train-epoch-loss:{train_epoch_loss}',
              f'val-epoch-loss:{val_epoch_loss}')

    # log min-loss of the model:
    wandb.log({'min-loss': min_loss})


if __name__ == '__main__':
    main()
