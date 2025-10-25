# dataset.py
# Contains the data loader for loading and preprocessing
import os
import glob
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset, DataLoader
import torch


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

    def __init__(self, data_dir, mode="train", transform=None):
        self.data_dir = data_dir
        self.transform = transform

        if mode == "train":
            image_folder = "keras_slices_train"
            mask_folder = "keras_slices_seg_train"
        elif mode == "validate":
            image_folder = "keras_slices_validate"
            mask_folder = "keras_slices_seg_validate"
        elif mode == "test":
            image_folder = "keras_slices_test"
            mask_folder = "keras_slices_seg_test"
        else:
            raise ValueError(
                f"Invalid mode '{mode}'. Choose 'train', 'validate', or 'test'."
            )

        image_dir = os.path.join(data_dir, image_folder)
        mask_dir = os.path.join(data_dir, mask_folder)

        self.image_paths = sorted(glob.glob(os.path.join(image_dir, "*.nii.gz")))
        self.mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.nii.gz")))

        if len(self.image_paths) == 0:
            print(f"Warning: No images found in {image_dir}")
        if len(self.mask_paths) == 0:
            print(f"Warning: No masks found in {mask_dir}")
        if len(self.image_paths) != len(self.mask_paths):
            print(f"Warning: Mismatch in number of images and masks!")

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
