import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

class SatelliteSegmentationDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_files = sorted(os.listdir(image_dir))
        self.label_files = sorted(os.listdir(label_dir))
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        label_path = os.path.join(self.label_dir, self.label_files[idx])

        image = np.array(Image.open(image_path).convert("RGB"))
        label = np.array(Image.open(label_path).convert("RGB"))[:, :, 0]  # class index

        if self.transform:
            augmented = self.transform(image=image, mask=label)
            image = augmented["image"]
            label = augmented["mask"].long()
        else:
            image = torch.tensor(image.transpose(2, 0, 1) / 255.0, dtype=torch.float32)
            label = torch.tensor(label, dtype=torch.long)

        return image, label
