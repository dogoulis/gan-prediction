#!/bin/sh

./scripts/lab_train.sh $1 Wang $2
./scripts/lab_train.sh $1 oneof $2
./scripts/lab_train.sh $1 strong $2
./scripts/lab_train.sh $1 soft $2
