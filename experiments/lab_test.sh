#!/bin/sh


# hold-out

# NOGIQA
#resnet
python evaluation.py -m resnet50 --test_dir ../data/test.csv --weights_dir ../checkpoints/second-exp/resnet/best-ckpt.pt --name resnet50 --project_name TEST-EXP
#vit-tiny
python evaluation.py -m vit-tiny --test_dir ../data/test.csv --weights_dir../checkpoints/second-exp/vit-tiny/best-ckpt.pt --name vit-tiny --project_name TEST-EXP
#swin-tiny
python evaluation.py -m swin-tiny --test_dir ../data/test.csv --weights_dir ../checkpoints/second-exp/swin-tiny/best-ckpt.pt --name swin-tiny --project_name TEST-EXP
#xception
python evaluation.py -m xception --test_dir ../data/test.csv --weights_dir ../checkpoints/second-exp/xception/best-ckpt.pt --name xception --project_name TEST-EXP



# GIQA
#resnet
python evaluation.py -m resnet50 --test_dir ../data/test.csv --weights_dir ../checkpoints/second-exp/resnet_giqa/best-ckpt.pt --name resnet_giqa --project_name TEST-EXP
#vit-tiny
python evaluation.py -m vit-tiny --test_dir ../data/test.csv --weights_dir ../checkpoints/second-exp/vit-tiny-giqa/best-ckpt.pt --name vit-tiny-giqa --project_name TEST-EXP
#swin-tiny
python evaluation.py -m swin-tiny --test_dir ../data/test.csv --weights_dir ../checkpoints/second-exp/swin_tiny_giqa/best-ckpt.pt --name swin-tiny-giqa --project_name TEST-EXP
#xception
python evaluation.py -m xception --test_dir ../data/test.csv --weights_dir ../checkpoints/second-exp/xception-giqa/best-ckpt.pt --name xception-giqa --project_name TEST-EXP


# progan ffhq

# NOGIQA
#resnet
python evaluation.py -m resnet50 --test_dir ../data/progan_test.csv --weights_dir ../checkpoints/second-exp/resnet/best-ckpt.pt --name resnet50-progan --project_name TEST-EXP
#vit-tiny
python evaluation.py -m vit-tiny --test_dir ../data/progan_test.csv --weights_dir ../checkpoints/second-exp/vit-tiny/best-ckpt.pt --name vit-tiny-progan --project_name TEST-EXP
#swin-tiny
python evaluation.py -m swin-tiny --test_dir ../data/progan_test.csv --weights_dir ../checkpoints/second-exp/swin-tiny/best-ckpt.pt --name swin-tiny-progan --project_name TEST-EXP
#xception
python evaluation.py -m xception --test_dir ../data/progan_test.csv --weights_dir ../checkpoints/second-exp/xception/best-ckpt.pt --name xception-progan --project_name TEST-EXP


# GIQA
#resnet
python evaluation.py -m resnet50 --test_dir ../data/progan_test.csv --weights_dir ../checkpoints/second-exp/resnet_giqa/best-ckpt.pt --name resnet_giqa-progan --project_name TEST-EXP
#vit-tiny
python evaluation.py -m vit-tiny --test_dir ../data/progan_test.csv --weights_dir ../checkpoints/second-exp/vit-tiny-giqa/best-ckpt.pt --name vit-tiny-giqa-progan --project_name TEST-EXP
#swin-tiny
python evaluation.py -m swin-tiny --test_dir ../data/progan_test.csv --weights_dir ../checkpoints/second-exp/swin_tiny_giqa/best-ckpt.pt --name swin-tiny-giqa-progan --project_name TEST-EXP
#xception
python evaluation.py -m xception --test_dir ../data/progan_test.csv --weights_dir ../checkpoints/second-exp/xception-giqa/best-ckpt.pt --name xception-giqa-progan --project_name TEST-EXP


# stylegan2 celeba

# NOGIQA
#resnet
python evaluation.py -m resnet50 --test_dir ../data/celeba_test.csv --weights_dir ../checkpoints/second-exp/resnet/best-ckpt.pt --name resnet50-celeba --project_name TEST-EXP
#vit-tiny
python evaluation.py -m vit-tiny --test_dir ../data/celeba_test.csv --weights_dir ../checkpoints/second-exp/vit-tiny/best-ckpt.pt --name vit-tiny-celeba --project_name TEST-EXP
#swin-tiny
python evaluation.py -m swin-tiny --test_dir ../data/celeba_test.csv --weights_dir ../checkpoints/second-exp/swin-tiny/best-ckpt.pt --name swin-tiny-celeba --project_name TEST-EXP
#xception
python evaluation.py -m xception --test_dir ../data/celeba_test.csv --weights_dir ../checkpoints/second-exp/xception/best-ckpt.pt --name xception-celeba --project_name TEST-EXP

# GIQA
#resnet
python evaluation.py -m resnet50 --test_dir ../data/celeba_test.csv --weights_dir ../checkpoints/second-exp/resnet_giqa/best-ckpt.pt --name resnet_giqa-celeba --project_name TEST-EXP
#vit-tiny
python evaluation.py -m vit-tiny --test_dir ../data/celeba_test.csv --weights_dir ../checkpoints/second-exp/vit-tiny-giqa/best-ckpt.pt --name vit-tiny-giqa-celeba --project_name TEST-EXP
#swin-tiny
python evaluation.py -m swin-tiny --test_dir ../data/celeba_test.csv --weights_dir ../checkpoints/second-exp/swin_tiny_giqa/best-ckpt.pt --name swin-tiny-giqa-celeba --project_name TEST-EXP
#xception
python evaluation.py -m xception --test_dir ../data/celeba_test.csv --weights_dir ../checkpoints/second-exp/xception-giqa/best-ckpt.pt --name xception-giqa-celeba --project_name TEST-EXP


# mixed

# NOGIQA
#resnet
python evaluation.py -m resnet50 --test_dir ../data/mixed_test.csv --weights_dir ../checkpoints/second-exp/resnet/best-ckpt.pt --name resnet50-mixed --project_name TEST-EXP
#vit-tiny
python evaluation.py -m vit-tiny --test_dir ../data/mixed_test.csv --weights_dir ../checkpoints/second-exp/vit-tiny/best-ckpt.pt --name vit-tiny-mixed --project_name TEST-EXP
#swin-tiny
python evaluation.py -m swin-tiny --test_dir ../data/mixed_test.csv --weights_dir ../checkpoints/second-exp/swin-tiny/best-ckpt.pt --name swin-tiny-mixed --project_name TEST-EXP
#xception
python evaluation.py -m xception --test_dir ../data/mixed_test.csv --weights_dir ../checkpoints/second-exp/xception/best-ckpt.pt --name xception-mixed --project_name TEST-EXP


# GIQA
#resnet
python evaluation.py -m resnet50 --test_dir ../data/mixed_test.csv --weights_dir ../checkpoints/second-exp/resnet_giqa/best-ckpt.pt --name resnet_giqa-mixed --project_name TEST-EXP
#vit-tiny
python evaluation.py -m vit-tiny --test_dir ../data/mixed_test.csv --weights_dir ../checkpoints/second-exp/vit-tiny-giqa/best-ckpt.pt --name vit-tiny-giqa-mixed --project_name TEST-EXP
#swin-tiny
python evaluation.py -m swin-tiny --test_dir ../data/mixed_test.csv --weights_dir ../checkpoints/second-exp/swin_tiny_giqa/best-ckpt.pt --name swin-tiny-giqa-mixed --project_name TEST-EXP
#xception
python evaluation.py -m xception --test_dir ../data/mixed_test.csv --weights_dir ../checkpoints/second-exp/xception-giqa/best-ckpt.pt --name xception-giqa-mixed --project_name TEST-EXP
