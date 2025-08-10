import numpy as np
import torch
from PIL import Image


class ImageInstanceSegmentationDataset:
    """
    A custom dataset class for image instance segmentation tasks.

    Args:
        data (list): A list of data samples, where each sample is a dictionary containing image and instance segmentation information.
        transforms (callable): A function/transform that takes in an image and returns transformed versions.
        config (object): A configuration object containing the column names for images, instance masks, bboxes, and categories.

    Attributes:
        data (list): The dataset containing image and instance segmentation information.
        transforms (callable): The transformation function to be applied to the images.
        config (object): The configuration object with image and target column names.

    Methods:
        __len__(): Returns the number of samples in the dataset.
        __getitem__(item): Retrieves the image and annotations at the specified index, applies transformations, and returns them as tensors.

    Example:
        dataset = ImageInstanceSegmentationDataset(data, transforms, config)
        batch = dataset[0]
    """

    def __init__(self, data, transforms, config):
        self.data = data
        self.transforms = transforms
        self.config = config

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        sample = self.data[item]
        
        # Get image
        image = sample[self.config.image_column]
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif hasattr(image, 'convert'):
            image = image.convert("RGB")
        
        # Convert image to numpy array
        image = np.array(image)
        
        # Get annotations
        annotations = {}
        
        # Instance masks
        if self.config.target_column in sample:
            instance_mask = sample[self.config.target_column]
            if hasattr(instance_mask, 'convert'):
                instance_mask = np.array(instance_mask.convert("L"))
            else:
                instance_mask = np.array(instance_mask)
            annotations["masks"] = instance_mask
        
        # Bounding boxes
        if self.config.bbox_column in sample:
            bboxes = sample[self.config.bbox_column]
            if isinstance(bboxes, list):
                bboxes = np.array(bboxes)
            annotations["boxes"] = bboxes
        
        # Categories/labels
        if self.config.category_column in sample:
            categories = sample[self.config.category_column]
            if isinstance(categories, list):
                categories = np.array(categories)
            annotations["labels"] = categories

        # Apply transforms if provided
        if self.transforms:
            # For instance segmentation, we primarily transform the image
            # Annotations might need special handling depending on the model
            image = self.transforms(image=image)["image"]

        # Convert image: (H, W, C) -> (C, H, W) and normalize to [0, 1]
        if len(image.shape) == 3:
            image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        else:
            image = image.astype(np.float32)

        # Prepare return dictionary
        result = {
            "pixel_values": torch.tensor(image, dtype=torch.float),
        }
        
        # Add annotations as labels for training
        if annotations:
            # Convert annotations to the format expected by the model
            labels = {}
            
            if "masks" in annotations:
                labels["masks"] = torch.tensor(annotations["masks"], dtype=torch.long)
            
            if "boxes" in annotations:
                labels["boxes"] = torch.tensor(annotations["boxes"], dtype=torch.float)
            
            if "labels" in annotations:
                labels["class_labels"] = torch.tensor(annotations["labels"], dtype=torch.long)
            
            result["labels"] = labels

        return result 