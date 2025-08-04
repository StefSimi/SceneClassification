import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from model import get_deeplabv3_model

# === Inference Dataset (no labels) ===
class InferenceDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_files = sorted(os.listdir(image_dir))
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        image = np.array(Image.open(image_path).convert("RGB"))

        if self.transform:
            image = self.transform(image=image)["image"]
        else:
            image = torch.tensor(image.transpose(2, 0, 1) / 255.0, dtype=torch.float32)

        return image, self.image_files[idx]

# === Transform ===
test_transform = A.Compose([
    A.Resize(512, 512),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])


# === Load model ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = get_deeplabv3_model(num_classes=9, pretrained=False).to(device)

# Load state dict but ignore aux_classifier if mismatched
state_dict = torch.load("best_model.pth", map_location=device)

# Filter out aux_classifier weights that don't match
filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith("aux_classifier.4")}
model.load_state_dict(filtered_state_dict, strict=False)
model.eval()


# === DataLoader ===
test_dataset = InferenceDataset("Data/images/test", transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# === Output folder ===
output_dir = "predicted_masks"
os.makedirs(output_dir, exist_ok=True)

# === Run inference ===
with torch.no_grad():
    for image, filename in test_loader:
        image = image.to(device)
        output = model(image)["out"]  # [1, C, H, W]
        pred_mask = torch.argmax(output.squeeze(), dim=0).cpu().numpy()  # [H, W]

        # Save predicted mask
        mask_img = Image.fromarray(pred_mask.astype(np.uint8))
        mask_img.save(os.path.join(output_dir, filename[0].replace('.tif', '_mask.png')))

print("Prediction complete. Masks saved in:", output_dir)