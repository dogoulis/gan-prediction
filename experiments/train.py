import os
import argparse
import numpy as np
from tqdm import tqdm

import timm
import torch
import torch.nn as nn
import wandb

from torch.utils.data.dataloader import DataLoader
from dataset import pytorch_dataset, augmentations

# parser:
parser = argparse.ArgumentParser(description='Training arguments')

parser.add_argument('--project_name', type=str, required=True,
                    metavar='project_name', help='Project name, utilized for logging purposes in W&B.')

parser.add_argument('-d', '--dataset_dir', type=str, required=True,
                    metavar='dataset_dir', help='Directory where the datasets are stored.')

parser.add_argument('-m', '--model', type=str,
                    metavar='model', help='which model to use in training: resnet50, swin-tiny, vit-tiny, xception')

parser.add_argument('-e', '--epochs', type=int, default=15,
                    metavar='epochs', help='Number of epochs')

parser.add_argument('-b', '--batch_size', type=int, default=32,
                    metavar='batch_size', help='input batch size for training (default: 32)')

parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3,
                    metavar='Learning Rate', help='learning rate of the optimizer (default: 1e-3)')

parser.add_argument('-wd', '--weight_decay', type=float, default=1e-5,
                    metavar='Weight Decay', help='Weight decay of the optimizer (default: 1e-5)')

parser.add_argument('--device', type=int, default=0,
                    metavar='device', help='device used during training (default: 0)')

parser.add_argument('--train_dir', type=str,
                    metavar='train-dir', help='training dataset path for csv')

parser.add_argument('--valid_dir', type=str,
                    metavar='valid-dir', help='validation dataset path for csv')

parser.add_argument('--save_dir', type=str,
                    metavar='save-dir', help='save directory path')

parser.add_argument('--name', type=str,
                    metavar='name', help='Experiment name that logs into wandb')

parser.add_argument('--group', type=str,
                    metavar='group', help='Grouping argument for W&B init.')

parser.add_argument('--workers', type=str, default=12,
                    metavar='workers', help='Number of workers for the dataloader')

parser.add_argument('--fp16', type=str, default=None,
                    metavar='fp16', help='Indicator for using mixed precision')

parser.add_argument('--aug', type=str, default='Wang',
                    metavar='aug', help='Indicator for employed augmentations')
args = parser.parse_args()


# define training logic
def train_epoch(model, train_dataloader, args, optimizer, criterion, scheduler=None,
                fp16_scaler=None, epoch=0, val_results={}):
    # to train only the classification layer:
    model.train()

    running_loss = []
    pbar = tqdm(train_dataloader, desc='epoch {}'.format(epoch), unit='iter')
    for batch, (x, y) in enumerate(pbar):

        x = x.to(args.device)
        y = y.to(args.device).unsqueeze(1)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            outputs = model(x)
            loss = criterion(outputs, y)

        if fp16_scaler is not None:
            fp16_scaler.scale(loss).backward()
            fp16_scaler.step(optimizer)
            fp16_scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss.append(loss.detach().cpu().numpy())

        # log mean loss for the last 10 batches:
        if (batch+1) % 10 == 0:
            wandb.log({'train-step-loss': np.mean(running_loss[-10:])})
            pbar.set_postfix(loss='{:.3f} ({:.3f})'.format(running_loss[-1], np.mean(running_loss)), **val_results)

    # change the position of the scheduler:
    # scheduler.step()

    train_loss = np.mean(running_loss)

    wandb.log({'train-epoch-loss': train_loss})

    return train_loss


# define validation logic
@torch.no_grad()
def validate_epoch(model, val_dataloader, args, criterion):
    model.eval()

    running_loss, y_true, y_pred = [], [], []
    for x, y in val_dataloader:
        x = x.to(args.device)
        y = y.to(args.device).unsqueeze(1)

        outputs = model(x)
        loss = criterion(outputs, y)

        # loss calculation over batch
        running_loss.append(loss.cpu().numpy())

        # accuracy calculation over batch
        outputs = torch.sigmoid(outputs)
        outputs = torch.round(outputs)
        y_true.append(y.cpu())
        y_pred.append(outputs.cpu())

    y_true = torch.cat(y_true, 0).numpy()
    y_pred = torch.cat(y_pred, 0).numpy()
    val_loss = np.mean(running_loss)
    wandb.log({'validation-loss': val_loss})
    acc = 100. * np.mean(y_true == y_pred)
    wandb.log({'validation-accuracy': acc})
    return {'val_acc': acc, 'val_loss': val_loss}


# MAIN def
def main():

    # initialize weights and biases:
    wandb.init(project=args.project_name, name=args.name,
               config=vars(args), group=args.group, save_code=True)

    # initialize model:
    if args.model == 'resnet50':
        model = timm.create_model('resnet50', pretrained=True, num_classes=1)
    elif args.model == 'swin-tiny':
        model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=1)
    elif args.model == 'swin-small':
        model = timm.create_model('swin_small_patch4_window7_224', pretrained=True, num_classes=1)
    elif args.model == 'vit-tiny':
        model = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=1)
    elif args.model == 'vit-small':
        model = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=1)
    elif args.model == 'xception':
        model = timm.create_model('xception', pretrained=True, num_classes=1)
    else:
        raise Exception('Model architecture not supported.')

    model = model.to(args.device)

    train_transforms = augmentations.get_training_augmentations(args.aug)
    valid_transforms = augmentations.get_validation_augmentations()

    # set the paths for training
    train_dataset = pytorch_dataset.dataset2(
        args.dataset_dir, args.train_dir, train_transforms)
    val_dataset = pytorch_dataset.dataset2(
        args.dataset_dir, args.valid_dir, valid_transforms)

    # defining data loaders:
    train_dataloader = DataLoader(
        train_dataset, num_workers=args.workers, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(
        val_dataset, num_workers=args.workers, batch_size=args.batch_size, shuffle=False)

    # setting the optimizer:
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # setting the scheduler:
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer=optimizer, step_size=5, gamma=0.1)

    criterion = nn.BCEWithLogitsLoss()

    fp16_scaler = None
    if args.fp16 is not None:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # directory:
    save_dir = args.save_dir
    print(save_dir)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # set value for min-loss:
    min_loss, val_results = float('inf'), {}
    print('Training starts...')
    for epoch in range(args.epochs):

        wandb.log({'epoch': epoch})
        train_epoch(model, train_dataloader=train_dataloader, args=args, optimizer=optimizer, criterion=criterion,
                    scheduler=scheduler, fp16_scaler=fp16_scaler, epoch=epoch, val_results=val_results)
        val_results = validate_epoch(model, val_dataloader=val_dataloader, args=args, criterion=criterion)

        if val_results['val_loss'] < min_loss:
            min_loss = val_results['val_loss'].copy()
            torch.save(model.state_dict(), os.path.join(
                save_dir, 'best-ckpt.pt'))


if __name__ == '__main__':
    main()
