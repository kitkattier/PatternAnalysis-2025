import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from modules import ImprovedUNet
from dataset import HipMRIDataset

# --- Hyperparameters ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-5  # Lowered learning rate for stability
BATCH_SIZE = 16
NUM_EPOCHS = 25
SAVE_PATH = "best_model.pth"

# --- data path ---
# on rangpur this is /home/groups/comp3710/HipMRI_Study_open/keras_slices_data
DATA_DIR = ""


# --- Stable Dice + BCE Loss ---
class DiceBCELoss(nn.Module):
    """
    A combined loss function of Dice Loss and Binary Cross-Entropy.
    More stable than DiceLoss alone.
    """

    def __init__(self, weight=0.5, smooth=1e-6):
        super(DiceBCELoss, self).__init__()
        self.weight = weight  # weight for dice loss
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()  # More stable than sigmoid + BCELoss

    def forward(self, pred_logits, target):
        # BCE Loss
        bce_loss = self.bce(pred_logits, target)

        # Dice Loss
        pred_prob = torch.sigmoid(pred_logits)

        # Flatten
        pred_prob = pred_prob.view(-1)
        target = target.view(-1)

        intersection = (pred_prob * target).sum()
        dice_score = (2.0 * intersection + self.smooth) / (
            pred_prob.sum() + target.sum() + self.smooth
        )
        dice_loss = 1 - dice_score

        # Combined loss
        combined_loss = bce_loss + (self.weight * dice_loss)
        return combined_loss


def dice_coefficient(pred_logits, target, smooth=1e-6):
    """
    Calculates the Dice similarity coefficient from logits.
    """
    pred_prob = torch.sigmoid(pred_logits)
    pred_mask = (pred_prob > 0.5).float()

    # Flatten
    pred_mask = pred_mask.view(-1)
    target = target.view(-1)

    intersection = (pred_mask * target).sum()
    target_sum = target.sum()
    pred_sum = pred_mask.sum()

    dice = (2.0 * intersection + smooth) / (pred_sum + target_sum + smooth)

    # Handle the case where the target mask is empty
    if target_sum == 0:
        # If pred is also empty, score is 1.0, otherwise it's 0.0
        return 1.0 if pred_sum == 0 else 0.0

    return dice.item()


def train_one_epoch(loader, model, optimizer, loss_fn):
    model.train()
    loop = tqdm(loader, desc="Training")
    total_loss = 0.0

    for images, masks in loop:
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        pred_logits = model(images)
        loss = loss_fn(pred_logits, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    return total_loss / len(loader)


def validate(loader, model, loss_fn, desc="Validation"):
    model.eval()
    total_loss = 0.0
    total_dice = 0.0

    with torch.no_grad():
        loop = tqdm(loader, desc=desc)
        for images, masks in loop:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            pred_logits = model(images)
            loss = loss_fn(pred_logits, masks)
            dice = dice_coefficient(pred_logits, masks)

            total_loss += loss.item()
            total_dice += dice
            loop.set_postfix(val_loss=loss.item(), dice=dice)

    avg_loss = total_loss / len(loader)
    avg_dice = total_dice / len(loader)
    return avg_loss, avg_dice


def plot_history(history):
    """
    Plots training history.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(history["train_loss"], label="Train Loss")
    ax1.plot(history["val_loss"], label="Validation Loss")
    ax1.set_title("Loss History")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()

    ax2.plot(history["val_dice"], label="Validation Dice")
    ax2.set_title("Validation Dice Score")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Dice Score")
    ax2.axhline(y=0.75, color="r", linestyle="--", label="Target Dice (0.75)")
    ax2.legend()

    plt.savefig("training_history.png")
    print("Saved training history plot to training_history.png")


def main():
    print(f"Using device: {DEVICE}")

    # Load datasets using the correct modes
    try:
        train_dataset = HipMRIDataset(data_dir=DATA_DIR, mode="train")
        val_dataset = HipMRIDataset(data_dir=DATA_DIR, mode="validate")
        test_dataset = HipMRIDataset(data_dir=DATA_DIR, mode="test")

        if len(train_dataset) == 0:
            print("Error: No training data found after filtering. Exiting.")
            return

    except FileNotFoundError:
        print(f"Error: Directory not found: {DATA_DIR}")
        print("Please set the correct DATA_DIR path in train.py")
        return

    # Create DataLoaders
    # Added num_workers to prevent CPU bottleneck on SLURM
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    print(f"Test set size: {len(test_dataset)}")

    model = ImprovedUNet(in_channels=1, out_channels=1).to(DEVICE)
    loss_fn = DiceBCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = {"train_loss": [], "val_loss": [], "val_dice": []}
    best_dice = -1.0

    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")

        train_loss = train_one_epoch(train_loader, model, optimizer, loss_fn)
        val_loss, val_dice = validate(val_loader, model, loss_fn, desc="Validation")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)

        print(
            f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}"
        )

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"Saved new best model with Dice: {best_dice:.4f} to {SAVE_PATH}")

    print(f"\nTraining complete. Best validation Dice: {best_dice:.4f}")
    plot_history(history)

    print("\nRunning final test on unseen data...")
    model.load_state_dict(torch.load(SAVE_PATH))  # Load best model
    test_loss, test_dice = validate(test_loader, model, loss_fn, desc="Testing")
    print(f"Test Loss: {test_loss:.4f}, Test Dice: {test_dice:.4f}")
    print(f"Target Dice score was 0.75. Your model achieved {test_dice:.4f}.")


if __name__ == "__main__":
    main()
