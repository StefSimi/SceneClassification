import os
from PIL import Image
import matplotlib.pyplot as plt

# Paths
image_dir = "Data/images/test"
mask_dir = "predicted_masks"

# List of test images
image_files = sorted(os.listdir(image_dir))

# Number of images to visualize (set this to a smaller number if needed)
#num_samples = min(5, len(image_files))
num_samples=len(image_files)

for i in range(0,num_samples,10):
    image_path = os.path.join(image_dir, image_files[i])
    mask_path = os.path.join(mask_dir, image_files[i].replace('.tif', '_mask.png'))

    if not os.path.exists(mask_path):
        print(f"Mask not found for {image_files[i]}, skipping...")
        continue

    image = Image.open(image_path)
    mask = Image.open(mask_path)

    # Plot side by side
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title(f"Image: {image_files[i]}")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap='tab10')  # Use colormap for better contrast
    plt.title("Predicted Mask")
    plt.axis("off")

    plt.tight_layout()
    plt.show()
