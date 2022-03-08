#!/bin/sh

BATCH_SZ=64

# NO GIQA
# resnet train:
python experiments/train.py -d ~/GAN_detection/ -e 15 -m resnet50 -b $BATCH_SZ --train_dir data/train.csv --valid_dir data/val.csv \
--save_dir checkpoints/second-exp/no_iso/resnet/ --project_name second-exp --name resnet50 -lr 1e-3

# swin-tiny train:
python experiments/train.py -d ~/GAN_detection/ -e 15 -m swin-tiny -b $BATCH_SZ --train_dir data/train.csv --valid_dir data/val.csv \
--save_dir checkpoints/second-exp/no_iso/swin_tiny/ --project_name second-exp --name swin-tiny -lr 1e-4

# vit-tiny train:
python experiments/train.py -d ~/GAN_detection/ -e 15 -m vit-tiny -b $BATCH_SZ --train_dir data/train.csv --valid_dir  data/val.csv \
--save_dir checkpoints/second-exp/no_iso/vit_tiny/ --project_name second-exp --name vit-tiny -lr 1e-4

# xception train:
python experiments/train.py -d ~/GAN_detection/ -e 15 -m xception -b $BATCH_SZ --train_dir data/train.csv --valid_dir data/val.csv \
--save_dir checkpoints/second-exp/no_iso/xception/ --project_name second-exp --name xception -lr 1e-3


# GIQA
# resnet train:
python experiments/train.py -d ~/GAN_detection/ -e 15 -m resnet50 -b $BATCH_SZ --train_dir data/traingiqa.csv --valid_dir data/valgiqa.csv \
--save_dir checkpoints/second-exp/no_iso/resnet_giqa/ --project_name second-exp --name resnet50-giqa -lr 1e-3

# swin-tiny train:
python experiments/train.py -d ~/GAN_detection/ -e 15 -m swin-tiny -b $BATCH_SZ --train_dir data/traingiqa.csv --valid_dir data/valgiqa.csv \
--save_dir checkpoints/second-exp/no_iso/swin_tiny_giqa/ --project_name second-exp --name swin-tiny-giqa -lr 1e-4

# vit-tiny train:
python experiments/train.py -d ~/GAN_detection/ -e 15 -m vit-tiny -b $BATCH_SZ --train_dir data/traingiqa.csv --valid_dir data/valgiqa.csv \
--save_dir checkpoints/second-exp/no_iso/vit_tiny_giqa/ --project_name second-exp --name vit-tiny-giqa -lr 1e-4

# xception train:
python experiments/train.py -d ~/GAN_detection/ -e 15 -m xception -b $BATCH_SZ --train_dir data/traingiqa.csv --valid_dir data/valgiqa.csv \
--save_dir checkpoints/second-exp/no_iso/xception_giqa/ --project_name second-exp --name xception-giqa -lr 1e-3


# ------------------------- ISO -------------------------
# NO GIQA
# resnet train:
python experiments/train.py -d ~/GAN_detection/ -e 15 -m resnet50 -b $BATCH_SZ --train_dir data/train.csv --valid_dir data/val.csv \
--save_dir checkpoints/second-exp/iso/resnet/ --project_name iso-second-exp --name resnet50 -lr 1e-3 --iso yes

# swin-tiny train:
python experiments/train.py -d ~/GAN_detection/ -e 15 -m swin-tiny -b $BATCH_SZ --train_dir data/train.csv --valid_dir data/val.csv \
--save_dir checkpoints/second-exp/iso/swin_tiny/ --project_name iso-second-exp --name swin-tiny -lr 1e-4 --iso yes

# vit-tiny train:
python experiments/train.py -d ~/GAN_detection/ -e 15 -m vit-tiny -b $BATCH_SZ --train_dir data/train.csv --valid_dir data/val.csv \
--save_dir checkpoints/second-exp/iso/vit_tiny/ --project_name iso-second-exp --name vit-tiny -lr 1e-4 --iso yes

# xception train:
python experiments/train.py -d ~/GAN_detection/ -e 15 -m xception -b $BATCH_SZ --train_dir data/train.csv --valid_dir data/val.csv \
--save_dir checkpoints/second-exp/iso/xception/ --project_name iso-second-exp --name xception -lr 1e-3 --iso yes


# GIQA
# resnet train:
python experiments/train.py -d ~/GAN_detection/ -e 15 -m resnet50 -b $BATCH_SZ --train_dir data/traingiqa.csv --valid_dir data/valgiqa.csv \
--save_dir checkpoints/second-exp/iso/resnet_giqa/ --project_name iso-second-exp --name resnet50-giqa -lr 1e-3 --iso yes

# swin-tiny train:
python experiments/train.py -d ~/GAN_detection/ -e 15 -m swin-tiny -b $BATCH_SZ --train_dir data/traingiqa.csv --valid_dir data/valgiqa.csv \
--save_dir checkpoints/second-exp/iso/swin_tiny_giqa/ --project_name iso-second-exp --name swin-tiny-giqa -lr 1e-4 --iso yes

# vit-tiny train:
python experiments/train.py -d ~/GAN_detection/ -e 15 -m vit-tiny -b $BATCH_SZ --train_dir data/traingiqa.csv --valid_dir data/valgiqa.csv \
--save_dir checkpoints/second-exp/iso/vit_tiny_giqa/ --project_name iso-second-exp --name vit-tiny-giqa -lr 1e-4 --iso yes

# xception train:
python experiments/train.py -d ~/GAN_detection/ -e 15 -m xception -b $BATCH_SZ --train_dir data/traingiqa.csv --valid_dir data/valgiqa.csv \
--save_dir checkpoints/second-exp/iso/xception_giqa/ --project_name second-exp --name xception-giqa -lr 1e-3 --iso yes
