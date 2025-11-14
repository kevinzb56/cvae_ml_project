import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class CelebADataset(Dataset):
    def __init__(self, csv_path, img_dir, attrs_list, transform=None):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform
        self.attrs_list = attrs_list
        # Replace -1 with 0 (not present) for all attributes
        self.df.replace(-1, 0, inplace=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx, 0]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        # Original images are 178x218. Crop to 178x178 center.
        w, h = image.size
        left = (w - 178) // 2
        top = (h - 178) // 2
        image = image.crop((left, top, left + 178, top + 178))

        if self.transform:
            image = self.transform(image)

        # Get the specific attributes we care about
        attrs = self.df.iloc[idx][self.attrs_list].values.astype(np.float32)
        
        return image, torch.from_numpy(attrs)

def get_dataloader(config):
    """
    Creates and returns the CelebA training dataloader.
    """
    transform = transforms.Compose([
        transforms.Resize(config.IMG_SIZE),
        transforms.CenterCrop(config.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Normalize to [-1, 1]
    ])

    dataset = CelebADataset(
        csv_path=config.CSV_PATH,
        img_dir=config.IMG_DIR,
        attrs_list=config.ATTRS,
        transform=transform
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=4, # Adjust based on your system
        pin_memory=True,
        drop_last=True
    )
    
    return dataloader