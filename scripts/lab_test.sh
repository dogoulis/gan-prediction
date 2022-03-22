#!/bin/sh

DATASET_DIR=~/GAN_detection/
BATCH_SZ=64
PROJECT_NAME=$1
GROUP=$2


# 30K

# NOGIQA
#resnet
python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m resnet50 --test_dir data/holdout.csv \
--weights_dir checkpoints/$PROJECT_NAME/resnet/30K/best-ckpt.pt --name resnet50-test --project_name SGD_test --group $GROUP
#vit-tiny
python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m vit-tiny --test_dir data/holdout.csv \
--weights_dir checkpoints/$PROJECT_NAME/vit_tiny/30K/best-ckpt.pt --name vit-tiny-test --project_name SGD_test --group $GROUP
#swin-tiny
python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m swin-tiny --test_dir data/holdout.csv \
--weights_dir checkpoints/$PROJECT_NAME/swin_tiny/30K/best-ckpt.pt --name swin-tiny-test --project_name SGD_test --group $GROUP
#xception
python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m xception --test_dir data/holdout.csv \
--weights_dir checkpoints/$PROJECT_NAME/xception/30K/best-ckpt.pt --name xception-test --project_name SGD_test --group $GROUP

# GIQA
#resnet
python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m resnet50 --test_dir data/holdout.csv \
--weights_dir checkpoints/$PROJECT_NAME/resnet_giqa/30K/best-ckpt.pt --name resnet50-test-giqa --project_name SGD_test --group $GROUP
#vit-tiny
python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m vit-tiny --test_dir data/holdout.csv \
--weights_dir checkpoints/$PROJECT_NAME/vit_tiny_giqa/30K/best-ckpt.pt --name vit-tiny-test-giqa --project_name SGD_test --group $GROUP
#swin-tiny
python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m swin-tiny --test_dir data/holdout.csv \
--weights_dir checkpoints/$PROJECT_NAME/swin_tiny_giqa/30K/best-ckpt.pt --name swin-tiny-test-giqa --project_name SGD_test --group $GROUP
#xception
python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m xception --test_dir data/holdout.csv \
--weights_dir checkpoints/$PROJECT_NAME/xception_giqa/30K/best-ckpt.pt --name xception-test-giqa --project_name SGD_test --group $GROUP


# progan ffhq

# NOGIQA
#resnet
python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m resnet50 --test_dir data/test_ffhq_progan.csv \
--weights_dir checkpoints/$PROJECT_NAME/resnet/30K/best-ckpt.pt --name resnet50-progan --project_name SGD_test --group $GROUP
#vit-tiny
python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m vit-tiny --test_dir data/test_ffhq_progan.csv \
--weights_dir checkpoints/$PROJECT_NAME/vit_tiny/30K/best-ckpt.pt --name vit-tiny-progan --project_name SGD_test --group $GROUP
#swin-tiny
python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m swin-tiny --test_dir data/test_ffhq_progan.csv \
--weights_dir checkpoints/$PROJECT_NAME/swin_tiny/30K/best-ckpt.pt --name swin-tiny-progan --project_name SGD_test --group $GROUP
#xception
python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m xception --test_dir data/test_ffhq_progan.csv \
--weights_dir checkpoints/$PROJECT_NAME/xception/30K/best-ckpt.pt --name xception-progan --project_name SGD_test --group $GROUP


# GIQA
#resnet
python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m resnet50 --test_dir data/test_ffhq_progan.csv \
--weights_dir checkpoints/$PROJECT_NAME/resnet_giqa/30K/best-ckpt.pt --name resnet50-giqa-progan --project_name SGD_test --group $GROUP
#vit-tiny
python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m vit-tiny --test_dir data/test_ffhq_progan.csv \
--weights_dir checkpoints/$PROJECT_NAME/vit_tiny_giqa/30K/best-ckpt.pt --name vit-tiny-giqa-progan --project_name SGD_test --group $GROUP
#swin-tiny
python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m swin-tiny --test_dir data/test_ffhq_progan.csv \
--weights_dir checkpoints/$PROJECT_NAME/swin_tiny_giqa/30K/best-ckpt.pt --name swin-tiny-giqa-progan --project_name SGD_test --group $GROUP
#xception
python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m xception --test_dir data/test_ffhq_progan.csv \
--weights_dir checkpoints/$PROJECT_NAME/xception_giqa/30K/best-ckpt.pt --name xception-giqa-progan --project_name SGD_test --group $GROUP


# stylegan2 celeba

# NOGIQA
#resnet
python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m resnet50 --test_dir data/test_celeba_stylegan2.csv \
--weights_dir checkpoints/$PROJECT_NAME/resnet/30K/best-ckpt.pt --name resnet50-celeba --project_name SGD_test --group $GROUP
#vit-tiny
python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m vit-tiny --test_dir data/test_celeba_stylegan2.csv \
--weights_dir checkpoints/$PROJECT_NAME/vit_tiny/30K/best-ckpt.pt --name vit-tiny-celeba --project_name SGD_test --group $GROUP
#swin-tiny
python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m swin-tiny --test_dir data/test_celeba_stylegan2.csv \
--weights_dir checkpoints/$PROJECT_NAME/swin_tiny/30K/best-ckpt.pt --name swin-tiny-celeba --project_name SGD_test --group $GROUP
#xception
python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m xception --test_dir data/test_celeba_stylegan2.csv \
--weights_dir checkpoints/$PROJECT_NAME/xception/30K/best-ckpt.pt --name xception-celeba --project_name SGD_test --group $GROUP

# GIQA
#resnet
python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m resnet50 --test_dir data/test_celeba_stylegan2.csv \
--weights_dir checkpoints/$PROJECT_NAME/resnet_giqa/30K/best-ckpt.pt --name resnet50-giqa-celeba --project_name SGD_test --group $GROUP
#vit-tiny
python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m vit-tiny --test_dir data/test_celeba_stylegan2.csv \
--weights_dir checkpoints/$PROJECT_NAME/vit_tiny_giqa/30K/best-ckpt.pt --name vit-tiny-giqa-celeba --project_name SGD_test --group $GROUP
#swin-tiny
python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m swin-tiny --test_dir data/test_celeba_stylegan2.csv \
--weights_dir checkpoints/$PROJECT_NAME/swin_tiny_giqa/30K/best-ckpt.pt --name swin-tiny-giqa-celeba --project_name SGD_test --group $GROUP
#xception
python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m xception --test_dir data/test_celeba_stylegan2.csv \
--weights_dir checkpoints/$PROJECT_NAME/xception_giqa/30K/best-ckpt.pt --name xception-giqa-celeba --project_name SGD_test --group $GROUP


# mixed

# NOGIQA
#resnet
python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m resnet50 --test_dir data/test_celeba_progan.csv \
--weights_dir checkpoints/$PROJECT_NAME/resnet/30K/best-ckpt.pt --name resnet50-mixed --project_name SGD_test --group $GROUP
#vit-tiny
python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m vit-tiny --test_dir data/test_celeba_progan.csv \
--weights_dir checkpoints/$PROJECT_NAME/vit_tiny/30K/best-ckpt.pt --name vit-tiny-mixed --project_name SGD_test --group $GROUP
#swin-tiny
python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m swin-tiny --test_dir data/test_celeba_progan.csv \
--weights_dir checkpoints/$PROJECT_NAME/swin_tiny/30K/best-ckpt.pt --name swin-tiny-mixed --project_name SGD_test --group $GROUP
#xception
python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m xception --test_dir data/test_celeba_progan.csv \
--weights_dir checkpoints/$PROJECT_NAME/xception/30K/best-ckpt.pt --name xception-mixed --project_name SGD_test --group $GROUP


# GIQA
#resnet
python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m resnet50 --test_dir data/test_celeba_progan.csv \
--weights_dir checkpoints/$PROJECT_NAME/resnet_giqa/30K/best-ckpt.pt --name resnet50-giqa-mixed --project_name SGD_test --group $GROUP
#vit-tiny
python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m vit-tiny --test_dir data/test_celeba_progan.csv \
--weights_dir checkpoints/$PROJECT_NAME/vit_tiny_giqa/30K/best-ckpt.pt --name vit-tiny-giqa-mixed --project_name SGD_test --group $GROUP
#swin-tiny
python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m swin-tiny --test_dir data/test_celeba_progan.csv \
--weights_dir checkpoints/$PROJECT_NAME/swin_tiny_giqa/30K/best-ckpt.pt --name swin-tiny-giqa-mixed --project_name SGD_test --group $GROUP
#xception
python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m xception --test_dir data/test_celeba_progan.csv \
--weights_dir checkpoints/$PROJECT_NAME/xception_giqa/30K/best-ckpt.pt --name xception-giqa-mixed --project_name SGD_test --group $GROUP


# 20K

# NOGIQA
#resnet
python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m resnet50 --test_dir data/holdout.csv \
--weights_dir checkpoints/$PROJECT_NAME/resnet/20K/best-ckpt.pt --name resnet50-test --project_name SGD_test --group $GROUP
#vit-tiny
python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m vit-tiny --test_dir data/holdout.csv \
--weights_dir checkpoints/$PROJECT_NAME/vit_tiny/20K/best-ckpt.pt --name vit-tiny-test --project_name SGD_test --group $GROUP
#swin-tiny
python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m swin-tiny --test_dir data/holdout.csv \
--weights_dir checkpoints/$PROJECT_NAME/swin_tiny/20K/best-ckpt.pt --name swin-tiny-test --project_name SGD_test --group $GROUP
#xception
python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m xception --test_dir data/holdout.csv \
--weights_dir checkpoints/$PROJECT_NAME/xception/20K/best-ckpt.pt --name xception-test --project_name SGD_test --group $GROUP

# GIQA
#resnet
python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m resnet50 --test_dir data/holdout.csv \
--weights_dir checkpoints/$PROJECT_NAME/resnet_giqa/20K/best-ckpt.pt --name resnet50-test-giqa --project_name SGD_test --group $GROUP
#vit-tiny
python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m vit-tiny --test_dir data/holdout.csv \
--weights_dir checkpoints/$PROJECT_NAME/vit_tiny_giqa/20K/best-ckpt.pt --name vit-tiny-test-giqa --project_name SGD_test --group $GROUP
#swin-tiny
python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m swin-tiny --test_dir data/holdout.csv \
--weights_dir checkpoints/$PROJECT_NAME/swin_tiny_giqa/20K/best-ckpt.pt --name swin-tiny-test-giqa --project_name SGD_test --group $GROUP
#xception
python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m xception --test_dir data/holdout.csv \
--weights_dir checkpoints/$PROJECT_NAME/xception_giqa/20K/best-ckpt.pt --name xception-test-giqa --project_name SGD_test --group $GROUP


# progan ffhq

# NOGIQA
#resnet
python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m resnet50 --test_dir data/test_ffhq_progan.csv \
--weights_dir checkpoints/$PROJECT_NAME/resnet/20K/best-ckpt.pt --name resnet50-progan --project_name SGD_test --group $GROUP
#vit-tiny
python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m vit-tiny --test_dir data/test_ffhq_progan.csv \
--weights_dir checkpoints/$PROJECT_NAME/vit_tiny/20K/best-ckpt.pt --name vit-tiny-progan --project_name SGD_test --group $GROUP
#swin-tiny
python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m swin-tiny --test_dir data/test_ffhq_progan.csv \
--weights_dir checkpoints/$PROJECT_NAME/swin_tiny/20K/best-ckpt.pt --name swin-tiny-progan --project_name SGD_test --group $GROUP
#xception
python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m xception --test_dir data/test_ffhq_progan.csv \
--weights_dir checkpoints/$PROJECT_NAME/xception/20K/best-ckpt.pt --name xception-progan --project_name SGD_test --group $GROUP


# GIQA
#resnet
python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m resnet50 --test_dir data/test_ffhq_progan.csv \
--weights_dir checkpoints/$PROJECT_NAME/resnet_giqa/20K/best-ckpt.pt --name resnet50-giqa-progan --project_name SGD_test --group $GROUP
#vit-tiny
python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m vit-tiny --test_dir data/test_ffhq_progan.csv \
--weights_dir checkpoints/$PROJECT_NAME/vit_tiny_giqa/20K/best-ckpt.pt --name vit-tiny-giqa-progan --project_name SGD_test --group $GROUP
#swin-tiny
python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m swin-tiny --test_dir data/test_ffhq_progan.csv \
--weights_dir checkpoints/$PROJECT_NAME/swin_tiny_giqa/20K/best-ckpt.pt --name swin-tiny-giqa-progan --project_name SGD_test --group $GROUP
#xception
python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m xception --test_dir data/test_ffhq_progan.csv \
--weights_dir checkpoints/$PROJECT_NAME/xception_giqa/20K/best-ckpt.pt --name xception-giqa-progan --project_name SGD_test --group $GROUP


# stylegan2 celeba

# NOGIQA
#resnet
python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m resnet50 --test_dir data/test_celeba_stylegan2.csv \
--weights_dir checkpoints/$PROJECT_NAME/resnet/20K/best-ckpt.pt --name resnet50-celeba --project_name SGD_test --group $GROUP
#vit-tiny
python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m vit-tiny --test_dir data/test_celeba_stylegan2.csv \
--weights_dir checkpoints/$PROJECT_NAME/vit_tiny/20K/best-ckpt.pt --name vit-tiny-celeba --project_name SGD_test --group $GROUP
#swin-tiny
python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m swin-tiny --test_dir data/test_celeba_stylegan2.csv \
--weights_dir checkpoints/$PROJECT_NAME/swin_tiny/20K/best-ckpt.pt --name swin-tiny-celeba --project_name SGD_test --group $GROUP
#xception
python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m xception --test_dir data/test_celeba_stylegan2.csv \
--weights_dir checkpoints/$PROJECT_NAME/xception/20K/best-ckpt.pt --name xception-celeba --project_name SGD_test --group $GROUP

# GIQA
#resnet
python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m resnet50 --test_dir data/test_celeba_stylegan2.csv \
--weights_dir checkpoints/$PROJECT_NAME/resnet_giqa/20K/best-ckpt.pt --name resnet50-giqa-celeba --project_name SGD_test --group $GROUP
#vit-tiny
python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m vit-tiny --test_dir data/test_celeba_stylegan2.csv \
--weights_dir checkpoints/$PROJECT_NAME/vit_tiny_giqa/20K/best-ckpt.pt --name vit-tiny-giqa-celeba --project_name SGD_test --group $GROUP
#swin-tiny
python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m swin-tiny --test_dir data/test_celeba_stylegan2.csv \
--weights_dir checkpoints/$PROJECT_NAME/swin_tiny_giqa/20K/best-ckpt.pt --name swin-tiny-giqa-celeba --project_name SGD_test --group $GROUP
#xception
python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m xception --test_dir data/test_celeba_stylegan2.csv \
--weights_dir checkpoints/$PROJECT_NAME/xception_giqa/20K/best-ckpt.pt --name xception-giqa-celeba --project_name SGD_test --group $GROUP


# mixed

# NOGIQA
#resnet
python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m resnet50 --test_dir data/test_celeba_progan.csv \
--weights_dir checkpoints/$PROJECT_NAME/resnet/20K/best-ckpt.pt --name resnet50-mixed --project_name SGD_test --group $GROUP
#vit-tiny
python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m vit-tiny --test_dir data/test_celeba_progan.csv \
--weights_dir checkpoints/$PROJECT_NAME/vit_tiny/20K/best-ckpt.pt --name vit-tiny-mixed --project_name SGD_test --group $GROUP
#swin-tiny
python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m swin-tiny --test_dir data/test_celeba_progan.csv \
--weights_dir checkpoints/$PROJECT_NAME/swin_tiny/20K/best-ckpt.pt --name swin-tiny-mixed --project_name SGD_test --group $GROUP
#xception
python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m xception --test_dir data/test_celeba_progan.csv \
--weights_dir checkpoints/$PROJECT_NAME/xception/20K/best-ckpt.pt --name xception-mixed --project_name SGD_test --group $GROUP


# GIQA
#resnet
python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m resnet50 --test_dir data/test_celeba_progan.csv \
--weights_dir checkpoints/$PROJECT_NAME/resnet_giqa/20K/best-ckpt.pt --name resnet50-giqa-mixed --project_name SGD_test --group $GROUP
#vit-tiny
python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m vit-tiny --test_dir data/test_celeba_progan.csv \
--weights_dir checkpoints/$PROJECT_NAME/vit_tiny_giqa/20K/best-ckpt.pt --name vit-tiny-giqa-mixed --project_name SGD_test --group $GROUP
#swin-tiny
python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m swin-tiny --test_dir data/test_celeba_progan.csv \
--weights_dir checkpoints/$PROJECT_NAME/swin_tiny_giqa/20K/best-ckpt.pt --name swin-tiny-giqa-mixed --project_name SGD_test --group $GROUP
#xception
python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m xception --test_dir data/test_celeba_progan.csv \
--weights_dir checkpoints/$PROJECT_NAME/xception_giqa/20K/best-ckpt.pt --name xception-giqa-mixed --project_name SGD_test --group $GROUP


# 10K

# NOGIQA
#resnet
python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m resnet50 --test_dir data/holdout.csv \
--weights_dir checkpoints/$PROJECT_NAME/resnet/10K/best-ckpt.pt --name resnet50-test --project_name SGD_test --group $GROUP
#vit-tiny
python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m vit-tiny --test_dir data/holdout.csv \
--weights_dir checkpoints/$PROJECT_NAME/vit_tiny/10K/best-ckpt.pt --name vit-tiny-test --project_name SGD_test --group $GROUP
#swin-tiny
python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m swin-tiny --test_dir data/holdout.csv \
--weights_dir checkpoints/$PROJECT_NAME/swin_tiny/10K/best-ckpt.pt --name swin-tiny-test --project_name SGD_test --group $GROUP
#xception
python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m xception --test_dir data/holdout.csv \
--weights_dir checkpoints/$PROJECT_NAME/xception/10K/best-ckpt.pt --name xception-test --project_name SGD_test --group $GROUP

# GIQA
#resnet
python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m resnet50 --test_dir data/holdout.csv \
--weights_dir checkpoints/$PROJECT_NAME/resnet_giqa/10K/best-ckpt.pt --name resnet50-test-giqa --project_name SGD_test --group $GROUP
#vit-tiny
python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m vit-tiny --test_dir data/holdout.csv \
--weights_dir checkpoints/$PROJECT_NAME/vit_tiny_giqa/10K/best-ckpt.pt --name vit-tiny-test-giqa --project_name SGD_test --group $GROUP
#swin-tiny
python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m swin-tiny --test_dir data/holdout.csv \
--weights_dir checkpoints/$PROJECT_NAME/swin_tiny_giqa/10K/best-ckpt.pt --name swin-tiny-test-giqa --project_name SGD_test --group $GROUP
#xception
python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m xception --test_dir data/holdout.csv \
--weights_dir checkpoints/$PROJECT_NAME/xception_giqa/10K/best-ckpt.pt --name xception-test-giqa --project_name SGD_test --group $GROUP


# progan ffhq

# NOGIQA
#resnet
python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m resnet50 --test_dir data/test_ffhq_progan.csv \
--weights_dir checkpoints/$PROJECT_NAME/resnet/10K/best-ckpt.pt --name resnet50-progan --project_name SGD_test --group $GROUP
#vit-tiny
python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m vit-tiny --test_dir data/test_ffhq_progan.csv \
--weights_dir checkpoints/$PROJECT_NAME/vit_tiny/10K/best-ckpt.pt --name vit-tiny-progan --project_name SGD_test --group $GROUP
#swin-tiny
python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m swin-tiny --test_dir data/test_ffhq_progan.csv \
--weights_dir checkpoints/$PROJECT_NAME/swin_tiny/10K/best-ckpt.pt --name swin-tiny-progan --project_name SGD_test --group $GROUP
#xception
python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m xception --test_dir data/test_ffhq_progan.csv \
--weights_dir checkpoints/$PROJECT_NAME/xception/10K/best-ckpt.pt --name xception-progan --project_name SGD_test --group $GROUP


# GIQA
#resnet
python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m resnet50 --test_dir data/test_ffhq_progan.csv \
--weights_dir checkpoints/$PROJECT_NAME/resnet_giqa/10K/best-ckpt.pt --name resnet50-giqa-progan --project_name SGD_test --group $GROUP
#vit-tiny
python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m vit-tiny --test_dir data/test_ffhq_progan.csv \
--weights_dir checkpoints/$PROJECT_NAME/vit_tiny_giqa/10K/best-ckpt.pt --name vit-tiny-giqa-progan --project_name SGD_test --group $GROUP
#swin-tiny
python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m swin-tiny --test_dir data/test_ffhq_progan.csv \
--weights_dir checkpoints/$PROJECT_NAME/swin_tiny_giqa/10K/best-ckpt.pt --name swin-tiny-giqa-progan --project_name SGD_test --group $GROUP
#xception
python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m xception --test_dir data/test_ffhq_progan.csv \
--weights_dir checkpoints/$PROJECT_NAME/xception_giqa/10K/best-ckpt.pt --name xception-giqa-progan --project_name SGD_test --group $GROUP


# stylegan2 celeba

# NOGIQA
#resnet
python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m resnet50 --test_dir data/test_celeba_stylegan2.csv \
--weights_dir checkpoints/$PROJECT_NAME/resnet/10K/best-ckpt.pt --name resnet50-celeba --project_name SGD_test --group $GROUP
#vit-tiny
python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m vit-tiny --test_dir data/test_celeba_stylegan2.csv \
--weights_dir checkpoints/$PROJECT_NAME/vit_tiny/10K/best-ckpt.pt --name vit-tiny-celeba --project_name SGD_test --group $GROUP
#swin-tiny
python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m swin-tiny --test_dir data/test_celeba_stylegan2.csv \
--weights_dir checkpoints/$PROJECT_NAME/swin_tiny/10K/best-ckpt.pt --name swin-tiny-celeba --project_name SGD_test --group $GROUP
#xception
python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m xception --test_dir data/test_celeba_stylegan2.csv \
--weights_dir checkpoints/$PROJECT_NAME/xception/10K/best-ckpt.pt --name xception-celeba --project_name SGD_test --group $GROUP

# GIQA
#resnet
python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m resnet50 --test_dir data/test_celeba_stylegan2.csv \
--weights_dir checkpoints/$PROJECT_NAME/resnet_giqa/10K/best-ckpt.pt --name resnet50-giqa-celeba --project_name SGD_test --group $GROUP
#vit-tiny
python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m vit-tiny --test_dir data/test_celeba_stylegan2.csv \
--weights_dir checkpoints/$PROJECT_NAME/vit_tiny_giqa/10K/best-ckpt.pt --name vit-tiny-giqa-celeba --project_name SGD_test --group $GROUP
#swin-tiny
python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m swin-tiny --test_dir data/test_celeba_stylegan2.csv \
--weights_dir checkpoints/$PROJECT_NAME/swin_tiny_giqa/10K/best-ckpt.pt --name swin-tiny-giqa-celeba --project_name SGD_test --group $GROUP
#xception
python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m xception --test_dir data/test_celeba_stylegan2.csv \
--weights_dir checkpoints/$PROJECT_NAME/xception_giqa/10K/best-ckpt.pt --name xception-giqa-celeba --project_name SGD_test --group $GROUP


# mixed

# NOGIQA
#resnet
python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m resnet50 --test_dir data/test_celeba_progan.csv \
--weights_dir checkpoints/$PROJECT_NAME/resnet/10K/best-ckpt.pt --name resnet50-mixed --project_name SGD_test --group $GROUP
#vit-tiny
python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m vit-tiny --test_dir data/test_celeba_progan.csv \
--weights_dir checkpoints/$PROJECT_NAME/vit_tiny/10K/best-ckpt.pt --name vit-tiny-mixed --project_name SGD_test --group $GROUP
#swin-tiny
python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m swin-tiny --test_dir data/test_celeba_progan.csv \
--weights_dir checkpoints/$PROJECT_NAME/swin_tiny/10K/best-ckpt.pt --name swin-tiny-mixed --project_name SGD_test --group $GROUP
#xception
python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m xception --test_dir data/test_celeba_progan.csv \
--weights_dir checkpoints/$PROJECT_NAME/xception/10K/best-ckpt.pt --name xception-mixed --project_name SGD_test --group $GROUP


# GIQA
#resnet
python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m resnet50 --test_dir data/test_celeba_progan.csv \
--weights_dir checkpoints/$PROJECT_NAME/resnet_giqa/10K/best-ckpt.pt --name resnet50-giqa-mixed --project_name SGD_test --group $GROUP
#vit-tiny
python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m vit-tiny --test_dir data/test_celeba_progan.csv \
--weights_dir checkpoints/$PROJECT_NAME/vit_tiny_giqa/10K/best-ckpt.pt --name vit-tiny-giqa-mixed --project_name SGD_test --group $GROUP
#swin-tiny
python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m swin-tiny --test_dir data/test_celeba_progan.csv \
--weights_dir checkpoints/$PROJECT_NAME/swin_tiny_giqa/10K/best-ckpt.pt --name swin-tiny-giqa-mixed --project_name SGD_test --group $GROUP
#xception
python experiments/evaluation.py -d $DATASET_DIR -b $BATCH_SZ -m xception --test_dir data/test_celeba_progan.csv \
--weights_dir checkpoints/$PROJECT_NAME/xception_giqa/10K/best-ckpt.pt --name xception-giqa-mixed --project_name SGD_test --group $GROUP
