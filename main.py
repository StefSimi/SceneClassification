# semantic_segmentation_project/src/main.py

import os
import dataset
from model import get_deeplabv3_model
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from torch import nn, optim

# === Set up your local dataset paths ===
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
DATA_DIR = os.path.join(BASE_DIR, 'Data')

TRAIN_IMG_DIR = os.path.join(DATA_DIR, 'images', 'train')
TRAIN_LABEL_DIR = os.path.join(DATA_DIR, 'label', 'train')
VAL_IMG_DIR = os.path.join(DATA_DIR, 'images', 'val')
VAL_LABEL_DIR = os.path.join(DATA_DIR, 'label', 'val')

# === Data Transforms ===
transform = A.Compose([
    A.Resize(512, 512),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# === Datasets and Dataloaders ===
train_dataset = dataset.SatelliteSegmentationDataset(TRAIN_IMG_DIR, TRAIN_LABEL_DIR, transform=transform)
val_dataset = dataset.SatelliteSegmentationDataset(VAL_IMG_DIR, VAL_LABEL_DIR, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# === Model Setup ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = get_deeplabv3_model(num_classes=9, pretrained=True).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# === Training Loop ===
i=0
EPOCHS = 5
print("Started training")
for epoch in range(EPOCHS):
    print("Epoch "+str(epoch))
    model.train()
    running_loss = 0.0
    i = 0
    for images, masks in train_loader:
        i += 1
        print(str(i)+ "/"+ str(len(train_loader)))
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)['out']
        #print("Mask min:", masks.min().item(), "max:", masks.max().item())
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{EPOCHS} - Training Loss: {avg_loss:.4f}")

    # Validation Loop
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)['out']

            loss = criterion(outputs, masks)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch {epoch+1}/{EPOCHS} - Validation Loss: {avg_val_loss:.4f}")

# Save the trained model to disk
torch.save(model.state_dict(), "best_model.pth")
print("Training complete. Model saved to best_model.pth")

