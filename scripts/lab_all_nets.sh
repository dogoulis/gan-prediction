#!/bin/sh

./scripts/lab_train.sh resnet50 $1 $2
./scripts/lab_train.sh vit-tiny $1 $2
./scripts/lab_train.sh swin-tiny $1 $2
./scripts/lab_train.sh xception $1 $2
