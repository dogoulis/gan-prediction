#!/bin/sh


# hold-out

# NOGIQA
#resnet
python experiments/evaluation.py -m resnet50 --test_dir data/test.csv --weights_dir checkpoints/second-exp/resnet/best-ckpt.pt --name resnet50-test --project_name TEST-EXP
#vit-tiny
python experiments/evaluation.py -m vit-tiny --test_dir data/test.csv --weights_dircheckpoints/second-exp/vit_tiny/best-ckpt.pt --name vit-tiny-test --project_name TEST-EXP
#swin-tiny
python experiments/evaluation.py -m swin-tiny --test_dir data/test.csv --weights_dir checkpoints/second-exp/swin_tiny/best-ckpt.pt --name swin-tiny-test --project_name TEST-EXP
#xception
python experiments/evaluation.py -m xception --test_dir data/test.csv --weights_dir checkpoints/second-exp/xception/best-ckpt.pt --name xception-test --project_name TEST-EXP



# GIQA
#resnet
python experiments/evaluation.py -m resnet50 --test_dir data/test.csv --weights_dir checkpoints/second-exp/resnet_giqa/best-ckpt.pt --name resnet50-test-giqa --project_name TEST-EXP
#vit-tiny
python experiments/evaluation.py -m vit-tiny --test_dir data/test.csv --weights_dir checkpoints/second-exp/vit_tiny_giqa/best-ckpt.pt --name vit-tiny-test-giqa --project_name TEST-EXP
#swin-tiny
python experiments/evaluation.py -m swin-tiny --test_dir data/test.csv --weights_dir checkpoints/second-exp/swin_tiny_giqa/best-ckpt.pt --name swin-tiny-test-giqa --project_name TEST-EXP
#xception
python experiments/evaluation.py -m xception --test_dir data/test.csv --weights_dir checkpoints/second-exp/xception_giqa/best-ckpt.pt --name xception-test-giqa --project_name TEST-EXP


# progan ffhq

# NOGIQA
#resnet
python experiments/evaluation.py -m resnet50 --test_dir data/progan_test.csv --weights_dir checkpoints/second-exp/resnet/best-ckpt.pt --name resnet50-progan --project_name TEST-EXP
#vit-tiny
python experiments/evaluation.py -m vit-tiny --test_dir data/progan_test.csv --weights_dir checkpoints/second-exp/vit_tiny/best-ckpt.pt --name vit-tiny-progan --project_name TEST-EXP
#swin-tiny
python experiments/evaluation.py -m swin-tiny --test_dir data/progan_test.csv --weights_dir checkpoints/second-exp/swin_tiny/best-ckpt.pt --name swin-tiny-progan --project_name TEST-EXP
#xception
python experiments/evaluation.py -m xception --test_dir data/progan_test.csv --weights_dir checkpoints/second-exp/xception/best-ckpt.pt --name xception-progan --project_name TEST-EXP


# GIQA
#resnet
python experiments/evaluation.py -m resnet50 --test_dir data/progan_test.csv --weights_dir checkpoints/second-exp/resnet_giqa/best-ckpt.pt --name resnet50-giqa-progan --project_name TEST-EXP
#vit-tiny
python experiments/evaluation.py -m vit-tiny --test_dir data/progan_test.csv --weights_dir checkpoints/second-exp/vit_tiny_giqa/best-ckpt.pt --name vit-tiny-giqa-progan --project_name TEST-EXP
#swin-tiny
python experiments/evaluation.py -m swin-tiny --test_dir data/progan_test.csv --weights_dir checkpoints/second-exp/swin_tiny_giqa/best-ckpt.pt --name swin-tiny-giqa-progan --project_name TEST-EXP
#xception
python experiments/evaluation.py -m xception --test_dir data/progan_test.csv --weights_dir checkpoints/second-exp/xception_giqa/best-ckpt.pt --name xception-giqa-progan --project_name TEST-EXP


# stylegan2 celeba

# NOGIQA
#resnet
python experiments/evaluation.py -m resnet50 --test_dir data/celeba_test.csv --weights_dir checkpoints/second-exp/resnet/best-ckpt.pt --name resnet50-celeba --project_name TEST-EXP
#vit-tiny
python experiments/evaluation.py -m vit-tiny --test_dir data/celeba_test.csv --weights_dir checkpoints/second-exp/vit_tiny/best-ckpt.pt --name vit-tiny-celeba --project_name TEST-EXP
#swin-tiny
python experiments/evaluation.py -m swin-tiny --test_dir data/celeba_test.csv --weights_dir checkpoints/second-exp/swin_tiny/best-ckpt.pt --name swin-tiny-celeba --project_name TEST-EXP
#xception
python experiments/evaluation.py -m xception --test_dir data/celeba_test.csv --weights_dir checkpoints/second-exp/xception/best-ckpt.pt --name xception-celeba --project_name TEST-EXP

# GIQA
#resnet
python experiments/evaluation.py -m resnet50 --test_dir data/celeba_test.csv --weights_dir checkpoints/second-exp/resnet_giqa/best-ckpt.pt --name resnet_giqa-celeba --project_name TEST-EXP
#vit-tiny
python experiments/evaluation.py -m vit-tiny --test_dir data/celeba_test.csv --weights_dir checkpoints/second-exp/vit_tiny_giqa/best-ckpt.pt --name vit-tiny-giqa-celeba --project_name TEST-EXP
#swin-tiny
python experiments/evaluation.py -m swin-tiny --test_dir data/celeba_test.csv --weights_dir checkpoints/second-exp/swin_tiny_giqa/best-ckpt.pt --name swin-tiny-giqa-celeba --project_name TEST-EXP
#xception
python experiments/evaluation.py -m xception --test_dir data/celeba_test.csv --weights_dir checkpoints/second-exp/xception_giqa/best-ckpt.pt --name xception-giqa-celeba --project_name TEST-EXP


# mixed

# NOGIQA
#resnet
python experiments/evaluation.py -m resnet50 --test_dir data/mixed_test.csv --weights_dir checkpoints/second-exp/resnet/best-ckpt.pt --name resnet50-mixed --project_name TEST-EXP
#vit-tiny
python experiments/evaluation.py -m vit-tiny --test_dir data/mixed_test.csv --weights_dir checkpoints/second-exp/vit_tiny/best-ckpt.pt --name vit-tiny-mixed --project_name TEST-EXP
#swin-tiny
python experiments/evaluation.py -m swin-tiny --test_dir data/mixed_test.csv --weights_dir checkpoints/second-exp/swin_tiny/best-ckpt.pt --name swin-tiny-mixed --project_name TEST-EXP
#xception
python experiments/evaluation.py -m xception --test_dir data/mixed_test.csv --weights_dir checkpoints/second-exp/xception/best-ckpt.pt --name xception-mixed --project_name TEST-EXP


# GIQA
#resnet
python experiments/evaluation.py -m resnet50 --test_dir data/mixed_test.csv --weights_dir checkpoints/second-exp/resnet_giqa/best-ckpt.pt --name resnet_giqa-mixed --project_name TEST-EXP
#vit-tiny
python experiments/evaluation.py -m vit-tiny --test_dir ../data/mixed_test.csv --weights_dir checkpoints/second-exp/vit_tiny_giqa/best-ckpt.pt --name vit-tiny-giqa-mixed --project_name TEST-EXP
#swin-tiny
python experiments/evaluation.py -m swin-tiny --test_dir data/mixed_test.csv --weights_dir checkpoints/second-exp/swin_tiny_giqa/best-ckpt.pt --name swin-tiny-giqa-mixed --project_name TEST-EXP
#xception
python experiments/evaluation.py -m xception --test_dir data/mixed_test.csv --weights_dir checkpoints/second-exp/xception_giqa/best-ckpt.pt --name xception-giqa-mixed --project_name TEST-EXP


# ------------------------- ISO -------------------------

# NOGIQA
#resnet
python experiments/evaluation.py -m resnet50 --test_dir data/test.csv --weights_dir checkpoints/second-exp/iso/resnet/best-ckpt.pt --name resnet50-test-iso --project_name TEST-EXP --group iso
#vit-tiny
python experiments/evaluation.py -m vit-tiny --test_dir data/test.csv --weights_dir checkpoints/second-exp/iso/vit_tiny/best-ckpt.pt --name vit-tiny-test-iso --project_name TEST-EXP --group iso
#swin-tiny
python experiments/evaluation.py -m swin-tiny --test_dir data/test.csv --weights_dir checkpoints/second-exp/iso/swin_tiny/best-ckpt.pt --name swin-tiny-test-iso --project_name TEST-EXP --group iso
#xception
python experiments/evaluation.py -m xception --test_dir data/test.csv --weights_dir checkpoints/second-exp/iso/xception/best-ckpt.pt --name xception-test-iso --project_name TEST-EXP --group iso



# GIQA
#resnet
python experiments/evaluation.py -m resnet50 --test_dir data/test.csv --weights_dir checkpoints/second-exp/iso/resnet_giqa/best-ckpt.pt --name resnet50-test-giqa-iso --project_name TEST-EXP --group iso
#vit-tiny
python experiments/evaluation.py -m vit-tiny --test_dir data/test.csv --weights_dir checkpoints/second-exp/iso/vit_tiny_giqa/best-ckpt.pt --name vit-tiny-test-giqa-iso --project_name TEST-EXP --group iso
#swin-tiny
python experiments/evaluation.py -m swin-tiny --test_dir data/test.csv --weights_dir checkpoints/second-exp/iso/swin_tiny_giqa/best-ckpt.pt --name swin-tiny-test-giqa-iso --project_name TEST-EXP --group iso
#xception
python experiments/evaluation.py -m xception --test_dir data/test.csv --weights_dir checkpoints/second-exp/xception_giqa/best-ckpt.pt --name xception-test-giqa-iso --project_name TEST-EXP --group iso


# progan ffhq

# NOGIQA
#resnet
python experiments/evaluation.py -m resnet50 --test_dir data/progan_test.csv --weights_dir checkpoints/second-exp/iso/resnet/best-ckpt.pt --name resnet50-progan-iso --project_name TEST-EXP --group iso
#vit-tiny
python experiments/evaluation.py -m vit-tiny --test_dir data/progan_test.csv --weights_dir checkpoints/second-exp/iso/vit_tiny/best-ckpt.pt --name vit-tiny-progan-iso --project_name TEST-EXP --group iso
#swin-tiny
python experiments/evaluation.py -m swin-tiny --test_dir data/progan_test.csv --weights_dir checkpoints/second-exp/iso/swin_tiny/best-ckpt.pt --name swin-tiny-progan-iso --project_name TEST-EXP --group iso
#xception
python experiments/evaluation.py -m xception --test_dir data/progan_test.csv --weights_dir checkpoints/second-exp/iso/xception/best-ckpt.pt --name xception-progan-iso --project_name TEST-EXP --group iso



# GIQA
#resnet
python experiments/evaluation.py -m resnet50 --test_dir data/progan_test.csv --weights_dir checkpoints/second-exp/iso/resnet_giqa/best-ckpt.pt --name resnet50-giqa-progan-iso --project_name TEST-EXP --group iso
#vit-tiny
python experiments/evaluation.py -m vit-tiny --test_dir data/progan_test.csv --weights_dir checkpoints/second-exp/iso/vit_tiny_giqa/best-ckpt.pt --name vit-tiny-giqa-progan-iso --project_name TEST-EXP --group iso
#swin-tiny
python experiments/evaluation.py -m swin-tiny --test_dir data/progan_test.csv --weights_dir checkpoints/second-exp/iso/swin_tiny_giqa/best-ckpt.pt --name swin-tiny-giqa-progan-iso --project_name TEST-EXP --group iso
#xception
python experiments/evaluation.py -m xception --test_dir data/progan_test.csv --weights_dir checkpoints/second-exp/iso/xception_giqa/best-ckpt.pt --name xception-giqa-progan-iso --project_name TEST-EXP --group iso

# stylegan2 celeba

# NOGIQA
#resnet
python experiments/evaluation.py -m resnet50 --test_dir data/celeba_test.csv --weights_dir checkpoints/second-exp/iso/resnet/best-ckpt.pt --name resnet50-celeba-iso --project_name TEST-EXP --group iso
#vit-tiny
python experiments/evaluation.py -m vit-tiny --test_dir data/celeba_test.csv --weights_dir checkpoints/second-exp/iso/vit_tiny/best-ckpt.pt --name vit-tiny-celeba-iso --project_name TEST-EXP --group iso
#swin-tiny
python experiments/evaluation.py -m swin-tiny --test_dir data/celeba_test.csv --weights_dir checkpoints/second-exp/iso/swin_tiny/best-ckpt.pt --name swin-tiny-celeba-iso --project_name TEST-EXP --group iso
#xception
python experiments/evaluation.py -m xception --test_dir data/celeba_test.csv --weights_dir checkpoints/second-exp/iso/xception/best-ckpt.pt --name xception-celeba-iso --project_name TEST-EXP --group iso

# GIQA
#resnet
python experiments/evaluation.py -m resnet50 --test_dir data/celeba_test.csv --weights_dir checkpoints/second-exp/iso/resnet_giqa/best-ckpt.pt --name resnet50-giqa-celeba-iso --project_name TEST-EX --group iso
#vit-tiny
python experiments/evaluation.py -m vit-tiny --test_dir data/celeba_test.csv --weights_dir checkpoints/second-exp/iso/vit_tiny_giqa/best-ckpt.pt --name vit-tiny-giqa-celeba-iso --project_name TEST-EXP --group iso
#swin-tiny
python experiments/evaluation.py -m swin-tiny --test_dir data/celeba_test.csv --weights_dir checkpoints/second-exp/iso/swin_tiny_giqa/best-ckpt.pt --name swin-tiny-giqa-celeba-iso --project_name TEST-EXP --group iso
#xception
python experiments/evaluation.py -m xception --test_dir data/celeba_test.csv --weights_dir checkpoints/second-exp/iso/xception_giqa/best-ckpt.pt --name xception-giqa-celeba-iso --project_name TEST-EXP --group iso


# mixed

# NOGIQA
#resnet
python experiments/evaluation.py -m resnet50 --test_dir data/mixed_test.csv --weights_dir checkpoints/second-exp/iso/resnet/best-ckpt.pt --name resnet50-mixed-iso --project_name TEST-EXP --group iso
#vit-tiny
python experiments/evaluation.py -m vit-tiny --test_dir data/mixed_test.csv --weights_dir checkpoints/second-exp/iso/vit_tiny/best-ckpt.pt --name vit-tiny-mixed-iso --project_name TEST-EXP --group iso
#swin-tiny
python experiments/evaluation.py -m swin-tiny --test_dir data/mixed_test.csv --weights_dir checkpoints/second-exp/iso/swin_tiny/best-ckpt.pt --name swin-tiny-mixed-iso --project_name TEST-EXP --group iso
#xception
python experiments/evaluation.py -m xception --test_dir data/mixed_test.csv --weights_dir checkpoints/second-exp/iso/xception/best-ckpt.pt --name xception-mixed-iso --project_name TEST-EXP --group iso



# GIQA
#resnet
python experiments/evaluation.py -m resnet50 --test_dir data/mixed_test.csv --weights_dir checkpoints/second-exp/iso/resnet_giqa/best-ckpt.pt --name resnet50-giqa-mixed-iso --project_name TEST-EXP --group iso
#vit-tiny
python experiments/evaluation.py -m vit-tiny --test_dir data/mixed_test.csv --weights_dir checkpoints/second-exp/iso/vit_tiny_giqa/best-ckpt.pt --name vit-tiny-giqa-mixed-iso --project_name TEST-EXP --group iso
#swin-tiny
python experiments/evaluation.py -m swin-tiny --test_dir data/mixed_test.csv --weights_dir checkpoints/second-exp/iso/swin_tiny_giqa/best-ckpt.pt --name swin-tiny-giqa-mixed-iso --project_name TEST-EXP --group iso
#xception
python experiments/evaluation.py -m xception --test_dir data/mixed_test.csv --weights_dir checkpoints/second-exp/iso/xception_giqa/best-ckpt.pt --name xception-giqa-mixed-iso --project_name TEST-EXP --group iso
