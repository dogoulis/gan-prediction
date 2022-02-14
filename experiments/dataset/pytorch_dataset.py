from torch.utils.data import Dataset, DataLoader
import torch
from PIL import Image
import pandas as pd
from albumentations.pytorch import ToTensorV2


class dataset2(Dataset):
    
    def __init__(self, dataset_path, transforms):
        
        self.dataset= pd.read_csv(dataset_path)
        self.imgs = self.dataset.image_path.values
        self.labels = self.dataset.label.values
        self.transforms = transforms

    def __len__(self):
        return len(self.imgs)


    def __getitem__(self,idx):
        
        image = Image.open(self.imgs[idx])
        label = self.labels[idx]
        label = self.labels[idx]
        if self.transforms:
            image = self.transforms(image)
        else:
            image = ToTensorV2(image)
        label = torch.tensor(label).float()

        return image, label