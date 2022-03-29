#!/bin/sh

DATASET_DIR=~/GAN_detection/
BATCH_SZ=128
EPOCHS=20
PROJECT_NAME=SyGeD
NETWORK=$1
AUG=$2
GPU=$3
SIZE=$4
WORKERS=12

# No GIQA
python experiments/train.py -d $DATASET_DIR -e $EPOCHS -m $NETWORK -b $BATCH_SZ --train_dir data/train_$SIZE.csv --valid_dir data/val.csv --workers $WORKERS \
--save_dir checkpoints/$PROJECT_NAME/$AUG/$NETWORK/$SIZE/ --project_name "$PROJECT_NAME-train" --name $NETWORK -lr 1e-4 --aug $AUG --group $AUG --device $GPU --fp16 true

python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m $NETWORK --test_dir data/holdout.csv \
--weights_dir checkpoints/$PROJECT_NAME/$AUG/$NETWORK/$SIZE/best-ckpt.pt --name "$NETWORK-test" --project_name "$PROJECT_NAME-test" --group $AUG

python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m $NETWORK --test_dir data/test_ffhq_progan.csv \
--weights_dir checkpoints/$PROJECT_NAME/$AUG/$NETWORK/$SIZE/best-ckpt.pt --name "$NETWORK-progan" --project_name "$PROJECT_NAME-test" --group $AUG

python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m $NETWORK --test_dir data/test_celeba_stylegan2.csv \
--weights_dir checkpoints/$PROJECT_NAME/$AUG/$NETWORK/$SIZE/best-ckpt.pt --name "$NETWORK-celeba" --project_name "$PROJECT_NAME-test" --group $AUG

python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m $NETWORK --test_dir data/test_celeba_progan.csv \
--weights_dir checkpoints/$PROJECT_NAME/$AUG/$NETWORK/$SIZE/best-ckpt.pt --name "$NETWORK-mixed" --project_name "$PROJECT_NAME-test" --group $AUG


# GIQA
python experiments/train.py -d $DATASET_DIR -e $EPOCHS -m $NETWORK -b $BATCH_SZ --train_dir data/train_$SIZE\_giqa.csv --valid_dir data/val.csv --workers $WORKERS \
--save_dir checkpoints/$PROJECT_NAME/$AUG/"$NETWORK-giqa"/$SIZE/ --project_name "$PROJECT_NAME-train" --name $NETWORK-giqa -lr 1e-4 --aug $AUG --group $AUG --device $GPU --fp16 true

python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m $NETWORK --test_dir data/holdout.csv \
--weights_dir checkpoints/$PROJECT_NAME/$AUG/"$NETWORK-giqa"/$SIZE/best-ckpt.pt --name "$NETWORK-giqa-test" --project_name "$PROJECT_NAME-test" --group $AUG

python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m $NETWORK --test_dir data/test_ffhq_progan.csv \
--weights_dir checkpoints/$PROJECT_NAME/$AUG/"$NETWORK-giqa"/$SIZE/best-ckpt.pt --name "$NETWORK-giqa-progan" --project_name "$PROJECT_NAME-test" --group $AUG

python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m $NETWORK --test_dir data/test_celeba_stylegan2.csv \
--weights_dir checkpoints/$PROJECT_NAME/$AUG/"$NETWORK-giqa"/$SIZE/best-ckpt.pt --name "$NETWORK-giqa-celeba" --project_name "$PROJECT_NAME-test" --group $AUG

python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m $NETWORK --test_dir data/test_celeba_progan.csv \
--weights_dir checkpoints/$PROJECT_NAME/$AUG/"$NETWORK-giqa"/$SIZE/best-ckpt.pt --name "$NETWORK-giqa-mixed" --project_name "$PROJECT_NAME-test" --group $AUG