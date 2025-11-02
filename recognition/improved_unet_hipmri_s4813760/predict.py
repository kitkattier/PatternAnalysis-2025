import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import random

from modules import ImprovedUNet
from dataset import HipMRIDataset, load_nifti_2d

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "best_model.pth"  # path to saved model
DATA_DIR = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data"
NUM_SAMPLES = 3  # Number of random samples to predict


def predict_and_visualize():
    print(f"Loading model from {MODEL_PATH}")

    # Initialize model and load weights
    model = ImprovedUNet(in_channels=1, out_channels=1).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # Load dataset to get some test images
    try:
        dataset = HipMRIDataset(data_dir=DATA_DIR)
        if len(dataset) == 0:
            return
    except FileNotFoundError:
        print(f"Error: Directory not found: {DATA_DIR}")
        print("Please set the correct DATA_DIR path in predict.py")
        return

    print(f"Visualizing {NUM_SAMPLES} random predictions...")

    fig, axes = plt.subplots(NUM_SAMPLES, 3, figsize=(15, 5 * NUM_SAMPLES))
    if NUM_SAMPLES == 1:
        axes = [axes]

    for i in range(NUM_SAMPLES):
        # Get a random sample
        idx = random.randint(0, len(dataset) - 1)
        image, mask = dataset[idx]  # These are already tensors

        # Prepare for model
        image_tensor = image.unsqueeze(0).to(DEVICE)  # Add batch dim

        with torch.no_grad():
            pred_tensor = model(image_tensor)
            pred_tensor = torch.sigmoid(pred_tensor)
            pred_mask = (pred_tensor > 0.5).cpu().numpy()

        # Convert to numpy for plotting
        image_np = image.squeeze().numpy()
        mask_np = mask.squeeze().numpy()
        pred_mask_np = pred_mask.squeeze()

        # Plot
        axes[i, 0].imshow(image_np, cmap="gray")
        axes[i, 0].set_title(f"Original Image (Sample {idx})")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(mask_np, cmap="gray")
        axes[i, 1].set_title("Ground Truth Mask")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(pred_mask_np, cmap="gray")
        axes[i, 2].set_title("Predicted Mask")
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.savefig("predictions.png")
    print("Saved prediction visualizations to predictions.png")


if __name__ == "__main__":
    predict_and_visualize()
