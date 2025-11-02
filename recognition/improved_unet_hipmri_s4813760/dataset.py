import os
import glob
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import torchvision.transforms.functional as TF
from torchvision import transforms


# --- This function is adapted from Appendix B [cite: 242-342] ---
def load_nifti_2d_slice(path):
    """
    Loads a 2D Nifti file as a numpy array.
    """
    niftiImage = nib.load(path)
    inImage = niftiImage.get_fdata(caching="unchanged")  # read disk only

    if len(inImage.shape) == 3:
        inImage = inImage[:, :, 0]  # sometimes extra dims in HipMRI_study data

    inImage = inImage.astype(np.float32)
    return inImage


class HipMRIDataset(Dataset):
    """
    PyTorch Dataset class for the HipMRI 2D slices.
    This version filters the dataset to only include slices
    that contain the target label.
    """

    def __init__(self, data_dir, mode, target_label=5, target_size=(256, 256)):
        self.data_dir = data_dir
        self.mode = mode
        self.target_label = target_label
        self.target_size = target_size

        # Define paths based on mode
        image_folder = f"keras_slices_{mode}"
        mask_folder = f"keras_slices_seg_{mode}"

        self.image_dir = os.path.join(data_dir, image_folder)
        self.mask_dir = os.path.join(data_dir, mask_folder)

        image_paths = sorted(glob.glob(os.path.join(self.image_dir, "*.nii.gz")))
        mask_paths = sorted(glob.glob(os.path.join(self.mask_dir, "*.nii.gz")))

        if len(image_paths) == 0 or len(mask_paths) == 0:
            print(
                f"Warning: No images or masks found in {self.image_dir} or {self.mask_dir}"
            )

        # Filter dataset to only include slices with the target label
        print(f"Scanning {mode} dataset for prostate (label {self.target_label})...")
        self.image_paths, self.mask_paths = self._filter_dataset(
            image_paths, mask_paths
        )
        print(f"Found {len(self.image_paths)} {mode} slices containing the prostate.")

    def _filter_dataset(self, image_paths, mask_paths):
        """
        Scans all masks and returns only the paths to images/masks
        that contain the target_label.
        """
        filtered_image_paths = []
        filtered_mask_paths = []

        for img_path, mask_path in tqdm(
            zip(image_paths, mask_paths),
            total=len(image_paths),
            desc=f"Filtering {self.mode} set",
        ):
            mask = load_nifti_2d_slice(mask_path)
            if np.any(mask == self.target_label):
                filtered_image_paths.append(img_path)
                filtered_mask_paths.append(mask_path)

        return filtered_image_paths, filtered_mask_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image and mask
        image = load_nifti_2d_slice(self.image_paths[idx])
        mask = load_nifti_2d_slice(self.mask_paths[idx])

        # --- Focus on Prostate Label (Label 5) ---
        mask = (mask == self.target_label).astype(np.float32)

        # Normalize image to [0, 1]
        if image.max() > 0:
            image = (image - image.min()) / (image.max() - image.min())

        # Add channel dimension
        image = np.expand_dims(image, axis=0)  # Shape: (1, H, W)
        mask = np.expand_dims(mask, axis=0)  # Shape: (1, H, W)

        # Convert to tensors
        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)

        # Resize
        image = TF.resize(image, self.target_size)

        # use BILINEAR resize and threshold to prevent mask deletion
        mask = TF.resize(
            mask, self.target_size, interpolation=transforms.InterpolationMode.BILINEAR
        )
        mask = (mask > 0.5).float()

        return image, mask


if __name__ == "__main__":
    # data dir, for rangpur its /home/groups/comp3710/HipMRI_Study_open/keras_slices_data
    DATA_DIR = ""

    try:
        print("Testing 'train' mode...")
        train_dataset = HipMRIDataset(data_dir=DATA_DIR, mode="train")

        if len(train_dataset) > 0:
            dataloader = torch.utils.data.DataLoader(
                train_dataset, batch_size=4, shuffle=True
            )
            images, masks = next(iter(dataloader))

            print(f"Batch image shape: {images.shape}")
            print(f"Batch mask shape: {masks.shape}")
            print(f"Mask values: {torch.unique(masks)}")
            print(f"Image min: {images.min()}, max: {images.max()}")
        else:
            print("No training data found after filtering.")

        print("\nTesting 'validate' mode...")
        val_dataset = HipMRIDataset(data_dir=DATA_DIR, mode="validate")
        print(f"Found {len(val_dataset)} validation images after filtering.")

    except FileNotFoundError:
        print(f"Error: Directory not found: {DATA_DIR}")
        print("Please set the correct DATA_DIR path in dataset.py")
