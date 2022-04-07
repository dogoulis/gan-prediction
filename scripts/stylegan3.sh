#!/bin/sh

DATASET_DIR=/fssd1/user-data/dogoulis/
BATCH_SZ=1
W_DIR=/fssd1/user-data/dogoulis/networks



# stylegan3 test
#resnet50
python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m resnet50 --test_dir data/test_stylegan3.csv \
--weights_dir $W_DIR/resnet50.pt --name resnet50-test --project_name eval_st3 --group s3
#vit-small
python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m vit-small --test_dir data/test_stylegan3.csv \
--weights_dir $W_DIR/vit-small.pt  --name vit-tiny-test --project_name eval_st3 --group s3
#swin-tiny
python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m swin-tiny --test_dir data/test_stylegan3.csv \
--weights_dir $W_DIR/swin-tiny.pt  --name swin-tiny-test --project_name eval_st3 --group s3

# stylegan3-u test
#resnet50
python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m resnet50 --test_dir data/test_stylegan3_u.csv \
--weights_dir $W_DIR/resnet50.pt --name resnet50-test --project_name eval_st3 --group s3-u
#vit-small
python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m vit-small --test_dir data/test_stylegan3_u.csv \
--weights_dir $W_DIR/vit-small.pt  --name vit-tiny-test --project_name eval_st3 --group s3-u
#swin-tiny
python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m swin-tiny --test_dir data/test_stylegan3_u.csv \
--weights_dir $W_DIR/swin-tiny.pt  --name swin-tiny-test --project_name eval_st3 --group s3-u