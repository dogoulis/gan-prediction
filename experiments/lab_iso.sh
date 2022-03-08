#!/bin/sh

# NO GIQA


# resnet train:
python train.py -e 15 -m resnet50 -b 24 --train_dir ../data/train.csv --valid_dir ../data/val.csv --save_dir ../checkpoints/second-exp/iso/resnet/ --project_name iso-second-exp --name resnet50 -lr 1e-3 --iso yes

# swin-tiny train:
python train.py -e 15 -m swin-tiny -b 24 --train_dir ../data/train.csv --valid_dir ../data/val.csv --save_dir ../checkpoints/second-exp/iso/swin-tiny/ --project_name iso-second-exp --name swin-tiny -lr 1e-4 --iso yes

# vit-tiny train:
python train.py -e 15 -m vit-tiny -b 24 --train_dir ../data/train.csv --valid_dir ../data/val.csv --save_dir ../checkpoints/second-exp/iso/vit-tiny/ --project_name iso-second-exp --name vit-tiny -lr 1e-4 --iso yes

# xception train:
python train.py -e 15 -m xception -b 24 --train_dir ../data/train.csv --valid_dir ../data/val.csv --save_dir ../checkpoints/second-exp/iso/xception/ --project_name iso-second-exp --name xception -lr 1e-3 --iso yes


# GIQA

# resnet train:
python train.py -e 15 -m resnet50 -b 24 --train_dir ../data/traingiqa.csv --valid_dir ../data/valgiqa.csv --save_dir ../checkpoints/second-exp/iso/resnet_giqa/ --project_name iso-second-exp --name resnet_giqa -lr 1e-3 --iso yes

# swin-tiny train:
python train.py -e 15 -m swin-tiny -b 24 --train_dir ../data/traingiqa.csv --valid_dir ../data/valgiqa.csv --save_dir ../checkpoints/second-exp/iso/swin_tiny_giqa/ --project_name iso-second-exp --name swin-tiny-giqa -lr 1e-4 --iso yes

# vit-tiny train:
python train.py -e 15 -m vit-tiny -b 24 --train_dir ../data/traingiqa.csv --valid_dir ../data/valgiqa.csv --save_dir ../checkpoints/second-exp/iso/vit-tiny-giqa/ --project_name iso-second-exp --name vit-tiny-giqa -lr 1e-4 --iso yes

# xception train:
python train.py -e 15 -m xception -b 24 --train_dir ../data/traingiqa.csv --valid_dir ../data/valgiqa.csv --save_dir ../checkpoints/second-exp/iso/xception-giqa/ --project_name second-exp --name xception-giqa -lr 1e-3 --iso yes
