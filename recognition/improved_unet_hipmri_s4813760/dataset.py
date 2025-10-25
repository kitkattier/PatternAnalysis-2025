# dataset.py
# Contains the data loader for loading and preprocessing
import os
import glob
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset, DataLoader
import torch

import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

# adapted from Appendix B
def load_nifti_2d(path):
    """
    Loads a 2D Nifti file as a numpy array.
    """
    niftiImage = nib.load(path)
    inImage = niftiImage.get_fdata(caching="unchanged")  # read disk only

    if len(inImage.shape) == 3:
        inImage = inImage[:, :, 0]  # sometimes extra dims in HipMRI_study data

    # Normalize image to [0, 1] for non-mask images
    inImage = inImage.astype(np.float32)
    if inImage.max() > 1.0:  # A simple check to see if it's a mask
        inImage = (inImage - inImage.min()) / (inImage.max() - inImage.min())

    return inImage


class HipMRIDataset(Dataset):
    """
    Dataset class for HipMRI 2D slices.
    """
    def __init__(self, data_dir, mode="train"):
        self.data_dir = data_dir

        self.image_dir = os.path.join(data_dir, f"keras_slices_{mode}")
        self.mask_dir = os.path.join(data_dir, f"keras_slices_seg_{mode}")

        self.image_paths = sorted(glob.glob(os.path.join(self.image_dir, "*.nii.gz")))
        self.mask_paths = sorted(glob.glob(os.path.join(self.mask_dir, "*.nii.gz")))

        self.mode = mode

        # define target size
        self.target_size = [256, 256]

        if len(self.image_paths) == 0 or len(self.mask_paths) == 0:
            print(f"Warning: No images or masks found in {self.image_dir} or {self.mask_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image and mask
        image = load_nifti_2d(self.image_paths[idx])
        mask = load_nifti_2d(self.mask_paths[idx])

        # focus on the prostate region (label 5)
        mask = (mask == 5).astype(np.float32)

        # add channel dimensions
        image = np.expand_dims(image, axis=0)
        mask = np.expand_dims(mask, axis=0)

        # convert to tensors
        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)

        image = TF.resize(image, self.target_size, interpolation=InterpolationMode.BILINEAR)
        mask = TF.resize(mask, self.target_size, interpolation=InterpolationMode.NEAREST)

        return image, mask


if __name__ == "__main__":
    DATA_DIR = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data"

    try:
        print("Testing 'train' split...")
        train_dataset = HipMRIDataset(data_dir=DATA_DIR, mode="train")
        print(f"Found {len(train_dataset)} training images/masks.")

        print("\nTesting 'validate' split...")
        val_dataset = HipMRIDataset(data_dir=DATA_DIR, mode="validate")
        print(f"Found {len(val_dataset)} validation images/masks.")

        print("\nTesting 'test' split...")
        test_dataset = HipMRIDataset(data_dir=DATA_DIR, mode="test")
        print(f"Found {len(test_dataset)} test images/masks.")

        if len(train_dataset) > 0:
            dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
            images, masks = next(iter(dataloader))

            print(f"\nBatch image shape: {images.shape}")
            print(f"Batch mask shape: {masks.shape}")
            print(f"Mask values: {torch.unique(masks)}")

    except FileNotFoundError:
        print(f"Error: Directory not found: {DATA_DIR}")
        print("Please set the correct DATA_DIR path in dataset.py")