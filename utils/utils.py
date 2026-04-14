# src/utils.py
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os
from torchvision import transforms

class FaceAttentionDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx]["filename"]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("L")  # Grayscale
        label = int(self.data.iloc[idx]["label"])

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)

def get_transform():
    return transforms.Compose([
        transforms.ToTensor(),  # [0,1] + (C,H,W)
    ])