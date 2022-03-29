#!/bin/sh

./scripts/lab_train.sh $1 Wang $2 $3
./scripts/lab_train.sh $1 oneof $2 $3
./scripts/lab_train.sh $1 strong $2 $3
./scripts/lab_train.sh $1 soft $2 $3

./scripts/lab_train.sh $1 Wang-geometric $2 $3
./scripts/lab_train.sh $1 oneof-geometric $2 $3
./scripts/lab_train.sh $1 strong-geometric $2 $3
./scripts/lab_train.sh $1 soft-geometric $2 $3