Experimental code for GAN human images detection.

In order to run train.py we have to provide the train dataset path of csv, i.e. a csv that contains image_paths and label for each image in the dataset. Those .csv files will be uploaded in the data folder.

## Script options
```
usage: train.py [-h] --project_name project_name [-d dataset_dir] [-m model] [-e epochs] [-b batch_size]
                [-lr Learning Rate] [-wd Weight Decay] [--device device] [--train_dir train-dir]
                [--valid_dir valid-dir] [--save_dir save-dir] [--name name] [--group group] [--workers workers]
                [--fp16 fp16] [--aug aug]

Training arguments

optional arguments:
  -h, --help            show this help message and exit
  --project_name project_name
                        Project name, utilized for logging purposes in W&B.
  -d dataset_dir, --dataset_dir dataset_dir
                        Directory where the datasets are stored. IT IS USED WITH A DIFFERENT DATA LOADER. CURRENTLY
                        USING --train_dir, --val_dir.
  -m model, --model model
                        which model to use in training: resnet50, swin-tiny, vit-tiny, xception
  -e epochs, --epochs epochs
                        Number of epochs
  -b batch_size, --batch_size batch_size
                        input batch size for training (default: 32)
  -lr Learning Rate, --learning_rate Learning Rate
                        learning rate of the optimizer (default: 1e-4)
  -wd Weight Decay, --weight_decay Weight Decay
                        Weight decay of the optimizer (default: 1e-2)
  --device device       device used during training (default: 0)
  --train_dir train-dir
                        training dataset path for csv
  --valid_dir valid-dir
                        validation dataset path for csv
  --save_dir save-dir   save directory path
  --name name           Experiment name that logs into wandb
  --group group         Grouping argument for W&B init.
  --workers workers     Number of workers for the dataloader
  --fp16 fp16           Indicator for using mixed precision
  --aug aug             Indicator for employed augmentations employed augmentations
```
