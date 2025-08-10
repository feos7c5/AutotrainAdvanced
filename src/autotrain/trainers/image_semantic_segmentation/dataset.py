import numpy as np
import torch


class ImageSemanticSegmentationDataset:
    """
    A custom dataset class for image semantic segmentation tasks.

    Args:
        data (list): A list of data samples, where each sample is a dictionary containing image and segmentation mask information.
        transforms (callable): A function/transform that takes in an image and mask and returns transformed versions.
        config (object): A configuration object containing the column names for images and segmentation masks.

    Attributes:
        data (list): The dataset containing image and segmentation mask information.
        transforms (callable): The transformation function to be applied to the images and masks.
        config (object): The configuration object with image and target column names.

    Methods:
        __len__(): Returns the number of samples in the dataset.
        __getitem__(item): Retrieves the image and mask at the specified index, applies transformations, and returns them as tensors.

    Example:
        dataset = ImageSemanticSegmentationDataset(data, transforms, config)
        image, mask = dataset[0]
    """

    def __init__(self, data, transforms, config):
        self.data = data
        self.transforms = transforms
        self.config = config

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        image = self.data[item][self.config.image_column]
        mask = self.data[item][self.config.target_column]

        # Convert image to RGB numpy array
        image = np.array(image.convert("RGB"))
        
        # Convert mask to numpy array and ensure it's grayscale
        if hasattr(mask, 'convert'):
            mask = np.array(mask.convert("L"))
        else:
            mask = np.array(mask)
        
        # Debug: Print unique values in first few samples
        if item < 3:
            unique_vals = np.unique(mask)
            print(f"Sample {item}: mask unique values = {unique_vals}, shape = {mask.shape}")
            for val in unique_vals:
                count = np.sum(mask == val)
                percentage = (count / mask.size) * 100
                print(f"  Value {val}: {count} pixels ({percentage:.1f}%)")
        
        # Apply albumentations transforms if provided
        if self.transforms:
            transformed = self.transforms(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        # Convert image: (H, W, C) -> (C, H, W) and normalize to [0, 1]
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)

        # Ensure mask is int64 for class indices
        mask = mask.astype(np.int64)
        
        # Apply reduce_labels if configured
        if hasattr(self.config, 'reduce_labels') and self.config.reduce_labels:
            # Reduce label IDs by 1 (common for datasets where 0 is background)
            mask = mask - 1
            # Set any negative values (original 0s) to 255 (ignore index)
            mask[mask < 0] = 255

        return {
            "pixel_values": torch.tensor(image, dtype=torch.float),
            "labels": torch.tensor(mask, dtype=torch.long),
        } 