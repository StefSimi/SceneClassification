# Satellite Image Segmentation with DeepLabV3

This project implements a semantic segmentation pipeline for satellite imagery using PyTorch and DeepLabV3 with a ResNet-50 backbone. The goal is to segment high-resolution landscape images into 8 distinct land cover classes:

- **Water**
- **Rangeland**
- **Developed Space**
- **Road**
- **Tree**
- **Bareland**
- **Agricultural Land**
- **Building**
---

## Key Features

-  **Custom Dataset Loader**: Efficiently reads satellite `.tif` images and grayscale-encoded segmentation masks, with automatic class index extraction.

-  **Model Architecture**: Uses `torchvision`'s `DeepLabV3_ResNet50` pretrained on ImageNet, fine-tuned for 9-class segmentation.

-  **Training Pipeline**:
  - Image resizing, normalization, and transformation via Albumentations.
  - Full training loop with validation, loss tracking, and GPU support.
  - Checkpoint saving (`best_model.pth`) after training.

-  **Inference Workflow**:
  - Runs model on test images.
  - Generates per-pixel class masks.
  - Saves output masks as `.png` files for downstream visualization.

-  **Visualization Tool**:
  - `visualize.py` displays side-by-side comparisons of test images and predicted segmentation masks.
  - Optional colormap support for human-friendly interpretation.

---





<img width="970" height="496" alt="image" src="https://github.com/user-attachments/assets/01118909-f6e5-4176-8e77-74574ae8921e" />
