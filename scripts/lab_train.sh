#!/bin/sh

DATASET_DIR=~/GAN_detection/
BATCH_SZ=64
EPOCHS=15
PROJECT_NAME=SyGeD
NETWORK=$1
AUG=$2
GPU=$3


# train 30k

# No GIQA
python experiments/train.py -d $DATASET_DIR -e $EPOCHS -m $NETWORK -b $BATCH_SZ --train_dir data/train_30k.csv --valid_dir data/val.csv \
--save_dir checkpoints/$PROJECT_NAME/$AUG/$NETWORK/30K/ --project_name "$PROJECT_NAME-train" --name $NETWORK -lr 1e-4 --aug $AUG --group $AUG --device $GPU

python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m $NETWORK --test_dir data/holdout.csv \
--weights_dir checkpoints/$PROJECT_NAME/$AUG/$NETWORK/30K/best-ckpt.pt --name "$NETWORK-test" --project_name "$PROJECT_NAME-test" --group $AUG

python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m $NETWORK --test_dir data/test_ffhq_progan.csv \
--weights_dir checkpoints/$PROJECT_NAME/$AUG/$NETWORK/30K/best-ckpt.pt --name "$NETWORK-progan" --project_name "$PROJECT_NAME-test" --group $AUG

python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m $NETWORK --test_dir data/test_celeba_stylegan2.csv \
--weights_dir checkpoints/$PROJECT_NAME/$AUG/$NETWORK/30K/best-ckpt.pt --name "$NETWORK-celeba" --project_name "$PROJECT_NAME-test" --group $AUG

python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m $NETWORK --test_dir data/test_celeba_progan.csv \
--weights_dir checkpoints/$PROJECT_NAME/$AUG/$NETWORK/30K/best-ckpt.pt --name "$NETWORK-mixed" --project_name "$PROJECT_NAME-test" --group $AUG


# GIQA
python experiments/train.py -d $DATASET_DIR -e $EPOCHS -m $NETWORK -b $BATCH_SZ --train_dir data/train_30k_giqa.csv --valid_dir data/valgiqa.csv \
--save_dir checkpoints/$PROJECT_NAME/$AUG/"$NETWORK-giqa"/30K/ --project_name "$PROJECT_NAME-train" --name $NETWORK-giqa -lr 1e-4 --aug $AUG --group $AUG --device $GPU

python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m $NETWORK --test_dir data/holdout.csv \
--weights_dir checkpoints/$PROJECT_NAME/$AUG/"$NETWORK-giqa"/30K/best-ckpt.pt --name "$NETWORK-giqa-test" --project_name "$PROJECT_NAME-test" --group $AUG

python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m $NETWORK --test_dir data/test_ffhq_progan.csv \
--weights_dir checkpoints/$PROJECT_NAME/$AUG/"$NETWORK-giqa"/30K/best-ckpt.pt --name "$NETWORK-giqa-progan" --project_name "$PROJECT_NAME-test" --group $AUG

python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m $NETWORK --test_dir data/test_celeba_stylegan2.csv \
--weights_dir checkpoints/$PROJECT_NAME/$AUG/"$NETWORK-giqa"/30K/best-ckpt.pt --name "$NETWORK-giqa-celeba" --project_name "$PROJECT_NAME-test" --group $AUG

python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m $NETWORK --test_dir data/test_celeba_progan.csv \
--weights_dir checkpoints/$PROJECT_NAME/$AUG/"$NETWORK-giqa"/30K/best-ckpt.pt --name "$NETWORK-giqa-mixed" --project_name "$PROJECT_NAME-test" --group $AUG

# train 20k

# No GIQA
python experiments/train.py -d $DATASET_DIR -e $EPOCHS -m $NETWORK -b $BATCH_SZ --train_dir data/train_20k.csv --valid_dir data/val.csv \
--save_dir checkpoints/$PROJECT_NAME/$AUG/$NETWORK/20K/ --project_name "$PROJECT_NAME-train" --name $NETWORK -lr 1e-4 --aug $AUG --group $AUG --device $GPU

python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m $NETWORK --test_dir data/holdout.csv \
--weights_dir checkpoints/$PROJECT_NAME/$AUG/$NETWORK/20K/best-ckpt.pt --name "$NETWORK-test" --project_name "$PROJECT_NAME-test" --group $AUG

python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m $NETWORK --test_dir data/test_ffhq_progan.csv \
--weights_dir checkpoints/$PROJECT_NAME/$AUG/$NETWORK/20K/best-ckpt.pt --name "$NETWORK-progan" --project_name "$PROJECT_NAME-test" --group $AUG

python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m $NETWORK --test_dir data/test_celeba_stylegan2.csv \
--weights_dir checkpoints/$PROJECT_NAME/$AUG/$NETWORK/20K/best-ckpt.pt --name "$NETWORK-celeba" --project_name "$PROJECT_NAME-test" --group $AUG

python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m $NETWORK --test_dir data/test_celeba_progan.csv \
--weights_dir checkpoints/$PROJECT_NAME/$AUG/$NETWORK/20K/best-ckpt.pt --name "$NETWORK-mixed" --project_name "$PROJECT_NAME-test" --group $AUG


# GIQA
python experiments/train.py -d $DATASET_DIR -e $EPOCHS -m $NETWORK -b $BATCH_SZ --train_dir data/train_20k_giqa.csv --valid_dir data/valgiqa.csv \
--save_dir checkpoints/$PROJECT_NAME/$AUG/"$NETWORK-giqa"/20K/ --project_name "$PROJECT_NAME-train" --name $NETWORK-giqa -lr 1e-4 --aug $AUG --group $AUG --device $GPU

python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m $NETWORK --test_dir data/holdout.csv \
--weights_dir checkpoints/$PROJECT_NAME/$AUG/"$NETWORK-giqa"/20K/best-ckpt.pt --name "$NETWORK-giqa-test" --project_name "$PROJECT_NAME-test" --group $AUG

python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m $NETWORK --test_dir data/test_ffhq_progan.csv \
--weights_dir checkpoints/$PROJECT_NAME/$AUG/"$NETWORK-giqa"/20K/best-ckpt.pt --name "$NETWORK-giqa-progan" --project_name "$PROJECT_NAME-test" --group $AUG

python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m $NETWORK --test_dir data/test_celeba_stylegan2.csv \
--weights_dir checkpoints/$PROJECT_NAME/$AUG/"$NETWORK-giqa"/20K/best-ckpt.pt --name "$NETWORK-giqa-celeba" --project_name "$PROJECT_NAME-test" --group $AUG

python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m $NETWORK --test_dir data/test_celeba_progan.csv \
--weights_dir checkpoints/$PROJECT_NAME/$AUG/"$NETWORK-giqa"/20K/best-ckpt.pt --name "$NETWORK-giqa-mixed" --project_name "$PROJECT_NAME-test" --group $AUG


# train 10k

# No GIQA
python experiments/train.py -d $DATASET_DIR -e $EPOCHS -m $NETWORK -b $BATCH_SZ --train_dir data/train_10k.csv --valid_dir data/val.csv \
--save_dir checkpoints/$PROJECT_NAME/$AUG/$NETWORK/10K/ --project_name "$PROJECT_NAME-train" --name $NETWORK -lr 1e-4 --aug $AUG --group $AUG --device $GPU

python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m $NETWORK --test_dir data/holdout.csv \
--weights_dir checkpoints/$PROJECT_NAME/$AUG/$NETWORK/10K/best-ckpt.pt --name "$NETWORK-test" --project_name "$PROJECT_NAME-test" --group $AUG

python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m $NETWORK --test_dir data/test_ffhq_progan.csv \
--weights_dir checkpoints/$PROJECT_NAME/$AUG/$NETWORK/10K/best-ckpt.pt --name "$NETWORK-progan" --project_name "$PROJECT_NAME-test" --group $AUG

python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m $NETWORK --test_dir data/test_celeba_stylegan2.csv \
--weights_dir checkpoints/$PROJECT_NAME/$AUG/$NETWORK/10K/best-ckpt.pt --name "$NETWORK-celeba" --project_name "$PROJECT_NAME-test" --group $AUG

python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m $NETWORK --test_dir data/test_celeba_progan.csv \
--weights_dir checkpoints/$PROJECT_NAME/$AUG/$NETWORK/10K/best-ckpt.pt --name "$NETWORK-mixed" --project_name "$PROJECT_NAME-test" --group $AUG


# GIQA
python experiments/train.py -d $DATASET_DIR -e $EPOCHS -m $NETWORK -b $BATCH_SZ --train_dir data/train_10k_giqa.csv --valid_dir data/valgiqa.csv \
--save_dir checkpoints/$PROJECT_NAME/$AUG/"$NETWORK-giqa"/10K/ --project_name "$PROJECT_NAME-train" --name $NETWORK-giqa -lr 1e-4 --aug $AUG --group $AUG --device $GPU

python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m $NETWORK --test_dir data/holdout.csv \
--weights_dir checkpoints/$PROJECT_NAME/$AUG/"$NETWORK-giqa"/10K/best-ckpt.pt --name "$NETWORK-giqa-test" --project_name "$PROJECT_NAME-test" --group $AUG

python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m $NETWORK --test_dir data/test_ffhq_progan.csv \
--weights_dir checkpoints/$PROJECT_NAME/$AUG/"$NETWORK-giqa"/10K/best-ckpt.pt --name "$NETWORK-giqa-progan" --project_name "$PROJECT_NAME-test" --group $AUG

python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m $NETWORK --test_dir data/test_celeba_stylegan2.csv \
--weights_dir checkpoints/$PROJECT_NAME/$AUG/"$NETWORK-giqa"/10K/best-ckpt.pt --name "$NETWORK-giqa-celeba" --project_name "$PROJECT_NAME-test" --group $AUG

python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m $NETWORK --test_dir data/test_celeba_progan.csv \
--weights_dir checkpoints/$PROJECT_NAME/$AUG/"$NETWORK-giqa"/10K/best-ckpt.pt --name "$NETWORK-giqa-mixed" --project_name "$PROJECT_NAME-test" --group $AUG