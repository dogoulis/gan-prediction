#!/bin/sh

DATASET_DIR=~/GAN_detection/
BATCH_SZ=64
EPOCHS=15
PROJECT_NAME=SyGeD
NETWORK=$1
AUG=$2
GPU=$3

# No GIQA
python experiments/train.py -d $DATASET_DIR -e $EPOCHS -m $NETWORK -b $BATCH_SZ --train_dir data/train.csv --valid_dir data/val.csv \
--save_dir checkpoints/$PROJECT_NAME/$AUG/$NETWORK/ --project_name "$PROJECT_NAME-train" --name $NETWORK -lr 1e-4 --aug $AUG --group $AUG --device $GPU

python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m $NETWORK --test_dir data/test.csv \
--weights_dir checkpoints/$PROJECT_NAME/$AUG/$NETWORK/best-ckpt.pt --name "$NETWORK-test" --project_name "$PROJECT_NAME-test" --group $AUG

python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m $NETWORK --test_dir data/progan_test.csv \
--weights_dir checkpoints/$PROJECT_NAME/$AUG/$NETWORK/best-ckpt.pt --name "$NETWORK-progan" --project_name "$PROJECT_NAME-test" --group $AUG

python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m $NETWORK --test_dir data/celeba_test.csv \
--weights_dir checkpoints/$PROJECT_NAME/$AUG/$NETWORK/best-ckpt.pt --name "$NETWORK-celeba" --project_name "$PROJECT_NAME-test" --group $AUG

python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m $NETWORK --test_dir data/mixed_test.csv \
--weights_dir checkpoints/$PROJECT_NAME/$AUG/$NETWORK/best-ckpt.pt --name "$NETWORK-mixed" --project_name "$PROJECT_NAME-test" --group $AUG


# GIQA
python experiments/train.py -d $DATASET_DIR -e $EPOCHS -m $NETWORK -b $BATCH_SZ --train_dir data/traingiqa.csv --valid_dir data/valgiqa.csv \
--save_dir checkpoints/$PROJECT_NAME/$AUG/"$NETWORK-giqa"/ --project_name "$PROJECT_NAME-train" --name $NETWORK-giqa -lr 1e-4 --aug $AUG --group $AUG --device $GPU

python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m $NETWORK --test_dir data/test.csv \
--weights_dir checkpoints/$PROJECT_NAME/$AUG/"$NETWORK-giqa"/best-ckpt.pt --name "$NETWORK-giqa-test" --project_name "$PROJECT_NAME-test" --group $AUG

python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m $NETWORK --test_dir data/progan_test.csv \
--weights_dir checkpoints/$PROJECT_NAME/$AUG/"$NETWORK-giqa"/best-ckpt.pt --name "$NETWORK-giqa-progan" --project_name "$PROJECT_NAME-test" --group $AUG

python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m $NETWORK --test_dir data/celeba_test.csv \
--weights_dir checkpoints/$PROJECT_NAME/$AUG/"$NETWORK-giqa"/best-ckpt.pt --name "$NETWORK-giqa-celeba" --project_name "$PROJECT_NAME-test" --group $AUG

python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m $NETWORK --test_dir data/mixed_test.csv \
--weights_dir checkpoints/$PROJECT_NAME/$AUG/"$NETWORK-giqa"/best-ckpt.pt --name "$NETWORK-giqa-mixed" --project_name "$PROJECT_NAME-test" --group $AUG
