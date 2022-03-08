#!/bin/sh

# NO GIQA


# resnet train:
python train.py -e 15 -m resnet50 -b 24 --train_dir ../data/train.csv --valid_dir ../data/val.csv --save_dir ../checkpoints/second-exp/resnet/ --project_name second-exp --name resnet50 -lr 1e-3 

# swin-tiny train:
python train.py -e 15 -m swin-tiny -b 24 --train_dir ../data/train.csv --valid_dir ../data/val.csv --save_dir ../checkpoints/second-exp/swin-tiny/ --project_name second-exp --name swin-tiny -lr 1e-4

# vit-tiny train:
python train.py -e 15 -m vit-tiny -b 24 --train_dir ../data/train.csv --valid_dir  ../data/val.csv --save_dir ../checkpoints/second-exp/vit-tiny/ --project_name second-exp --name vit-tiny -lr 1e-4

# xception train:
python train.py -e 15 -m xception -b 24 --train_dir ../data/train.csv --valid_dir ../data/val.csv --save_dir ../checkpoints/second-exp/xception/ --project_name second-exp --name xception -lr 1e-3


# GIQA

# resnet train:
python train.py -e 15 -m resnet50 -b 24 --train_dir ../data/traingiqa.csv --valid_dir ../data/valgiqa.csv --save_dir ../checkpoints/second-exp/resnet_giqa/ --project_name second-exp --name resnet_giqa -lr 1e-3 

# swin-tiny train:
python train.py -e 15 -m swin-tiny -b 24 --train_dir ../data/traingiqa.csv --valid_dir ../data/valgiqa.csv --save_dir ../checkpoints/second-exp/swin_tiny_giqa/ --project_name second-exp --name swin-tiny-giqa -lr 1e-4

# vit-tiny train:
python train.py -e 15 -m vit-tiny -b 24 --train_dir ../data/traingiqa.csv --valid_dir ../data/valgiqa.csv --save_dir ../checkpoints/second-exp/vit-tiny-giqa/ --project_name second-exp --name vit-tiny-giqa -lr 1e-4

# xception train:
python train.py -e 15 -m xception -b 24 --train_dir ../data/traingiqa.csv --valid_dir ../data/valgiqa.csv --save_dir ../checkpoints/second-exp/xception-giqa/ --project_name second-exp --name xception-giqa -lr 1e-3
