#!/bin/sh

DATASET_DIR=/fssd1/user-data/dogoulis/
BATCH_SZ=64
W_DIR=/fssd1/user-data/dogoulis/networks



# 5k giqa first
#resnet50
python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m resnet50 --test_dir data/test_first_5k.csv \
--weights_dir $W_DIR/resnet50.pt --name resnet50-test --project_name evaluation_giqa --group first_5k
#swin-tiny
python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m vit-tiny --test_dir data/test_first_5k.csv \
--weights_dir $W_DIR/swin-tiny.pt  --name vit-tiny-test --project_name evaluation_giqa --group first_5k
#vit-small
python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m swin-tiny --test_dir data/test_first_5k.csv \
--weights_dir $W_DIR/vit-small.pt  --name swin-tiny-test --project_name evaluation_giqa --group first_5k


# 5k giqa last
#resnet50
python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m resnet50 --test_dir data/test_last_5k.csv \
--weights_dir $W_DIR/resnet50.pt --name resnet50-test --project_name evaluation_giqa --group last_5k
#swin-tiny
python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m vit-tiny --test_dir data/test_last_5k.csv \
--weights_dir $W_DIR/swin-tiny.pt  --name vit-tiny-test --project_name evaluation_giqa --group last_5k
#vit-small
python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m swin-tiny --test_dir data/test_last_5k.csv \
--weights_dir $W_DIR/vit-small.pt  --name swin-tiny-test --project_name evaluation_giqa --group last_5k


# 2k giqa first
#resnet50
python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m resnet50 --test_dir data/test_first_2k.csv \
--weights_dir $W_DIR/resnet50.pt --name resnet50-test --project_name evaluation_giqa --group first_2k
#swin-tiny
python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m vit-tiny --test_dir data/test_first_2k.csv \
--weights_dir $W_DIR/swin-tiny.pt  --name vit-tiny-test --project_name evaluation_giqa --group first_2k
#vit-small
python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m swin-tiny --test_dir data/test_first_2k.csv \
--weights_dir $W_DIR/vit-small.pt  --name swin-tiny-test --project_name evaluation_giqa --group first_2k


# 2k giqa last
#resnet50
python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m resnet50 --test_dir data/test_last_2k.csv \
--weights_dir $W_DIR/resnet50.pt --name resnet50-test --project_name evaluation_giqa --group last_2k
#swin-tiny
python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m vit-tiny --test_dir data/test_last_2k.csv \
--weights_dir $W_DIR/swin-tiny.pt  --name vit-tiny-test --project_name evaluation_giqa --group last_2k
#vit-small
python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m swin-tiny --test_dir data/test_last_2k.csv \
--weights_dir $W_DIR/vit-small.pt  --name swin-tiny-test --project_name evaluation_giqa --group last_2k