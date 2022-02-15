Experimental code for GAN human images detection.

## Script options
```
usage: train.py [-h] [-m model] [-e epochs] [-b batch_size]
                [-lr Learning Rate] [-wd Weight Decay] [-gq GIQA] [-d device]
                [--train_dir train-dir] [--valid_dir valid-dir]
                [--save_dir save-dir]

Training arguments

optional arguments:
  -h, --help            show this help message and exit
  -m model, --model model
                        which model to use in training: resnet50, vit-large,
                        vit-base
  -e epochs, --epochs epochs
                        Number of epochs
  -b batch_size, --batch_size batch_size
                        input batch size for training (default: 32)
  -lr Learning Rate, --learning_rate Learning Rate
                        learning rate of the optimizer
  -wd Weight Decay, --weight_decay Weight Decay
                        Weight decay of the optimizer
  -gq GIQA, --giqa GIQA
                        Train with Giqa
  -d device, --device device
                        device used during training
  --train_dir train-dir
                        training dataset path for csv
  --valid_dir valid-dir
                        validation dataset path for csv
  --save_dir save-dir   save directory path
```
