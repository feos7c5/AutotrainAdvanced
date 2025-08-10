import os

import albumentations as A
import numpy as np
import torch
from sklearn import metrics

from autotrain.trainers.image_instance_segmentation.dataset import ImageInstanceSegmentationDataset


INSTANCE_SEGMENTATION_EVAL_METRICS = (
    "eval_loss",
    "eval_bbox_map",
    "eval_segm_map",
)

MODEL_CARD = """
---
tags:
- autotrain
- transformers
- image-segmentation
- instance-segmentation{base_model}
widget:
- src: https://huggingface.co/datasets/mishig/sample_images/resolve/main/tiger.jpg
  example_title: Tiger
- src: https://huggingface.co/datasets/mishig/sample_images/resolve/main/teapot.jpg
  example_title: Teapot
- src: https://huggingface.co/datasets/mishig/sample_images/resolve/main/palace.jpg
  example_title: Palace{dataset_tag}
---

# Model Trained Using AutoTrain

- Problem type: Image Instance Segmentation
"""


def compute_metrics(eval_pred):
    """Compute metrics for evaluation."""
    predictions, labels = eval_pred
    
    # For instance segmentation, metrics are typically computed during inference
    # This is a placeholder that can be extended based on the specific model requirements
    
    # Basic loss-based metric
    if hasattr(predictions, 'loss') and predictions.loss is not None:
        return {"eval_loss": float(predictions.loss)}
    
    # If predictions contain logits, we can compute some basic metrics
    if hasattr(predictions, 'logits'):
        # This is a simplified metric computation
        # In practice, you'd want more sophisticated metrics like mAP for bounding boxes and masks
        return {"eval_loss": 0.0}
    
    return {"eval_loss": 0.0}


def process_data(train_data, valid_data, image_processor, config):
    """
    Processes training and validation data for image instance segmentation.

    Args:
        train_data (Dataset): The training dataset.
        valid_data (Dataset or None): The validation dataset. Can be None if no validation data is provided.
        image_processor (ImageProcessor): An object containing image processing parameters such as size, mean, and std.
        config (dict): Configuration dictionary containing additional parameters for dataset processing.

    Returns:
        tuple: A tuple containing the processed training dataset and the processed validation dataset (or None if no validation data is provided).
    """
    # Get image size from processor
    if hasattr(image_processor, 'size'):
        if isinstance(image_processor.size, dict):
            if "shortest_edge" in image_processor.size:
                size = image_processor.size["shortest_edge"]
            elif "height" in image_processor.size and "width" in image_processor.size:
                size = (image_processor.size["height"], image_processor.size["width"])
            else:
                size = 512  # Default size
        else:
            size = image_processor.size
    else:
        size = 512  # Default size
    
    try:
        height, width = size
    except (TypeError, ValueError):
        height = size
        width = size

    # Get normalization parameters
    mean = getattr(image_processor, 'image_mean', [0.485, 0.456, 0.406])
    std = getattr(image_processor, 'image_std', [0.229, 0.224, 0.225])

    train_transforms = A.Compose(
        [
            A.Resize(height=height, width=width),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=mean, std=std),
        ],
        is_check_shapes=False
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=height, width=width),
            A.Normalize(mean=mean, std=std),
        ],
        is_check_shapes=False
    )

    train_data = ImageInstanceSegmentationDataset(train_data, train_transforms, config)
    if valid_data is not None:
        valid_data = ImageInstanceSegmentationDataset(valid_data, val_transforms, config)
        return train_data, valid_data
    return train_data, None


def create_model_card(config, trainer, num_classes):
    """
    Generates a model card for the given configuration and trainer.

    Args:
        config (object): Configuration object containing various settings.
        trainer (object): Trainer object used for model training and evaluation.
        num_classes (int): Number of classes in the instance segmentation task.

    Returns:
        str: The generated model card as a string.
    """
    if os.path.exists(f"{config.project_name}/README.md"):
        return
    
    model_card = MODEL_CARD

    if config.data_path:
        model_card = model_card.replace("{dataset_tag}", f"\ndatasets:\n- {config.data_path}")
    else:
        model_card = model_card.replace("{dataset_tag}", "")

    if config.model:
        model_card = model_card.replace("{base_model}", f"\nbase_model: {config.model}")
    else:
        model_card = model_card.replace("{base_model}", "")

    eval_results = ""
    if trainer.state.log_history:
        eval_results = "## Training Results\n\n"
        for log in trainer.state.log_history:
            if "eval_loss" in log:
                eval_results += f"- Eval Loss: {log['eval_loss']:.4f}\n"
            if "eval_bbox_map" in log:
                eval_results += f"- Eval BBox mAP: {log['eval_bbox_map']:.4f}\n"
            if "eval_segm_map" in log:
                eval_results += f"- Eval Segmentation mAP: {log['eval_segm_map']:.4f}\n"
            if "train_loss" in log:
                eval_results += f"- Train Loss: {log['train_loss']:.4f}\n"
    
    model_card += f"\n\n{eval_results}"
    model_card += f"\n\n## Model Details\n\n"
    model_card += f"- Problem Type: Image Instance Segmentation\n"
    model_card += f"- Model Architecture: {config.model}\n"
    model_card += f"- Number of Classes: {num_classes}\n"
    model_card += f"- Training Epochs: {config.epochs}\n"
    model_card += f"- Batch Size: {config.batch_size}\n"
    model_card += f"- Learning Rate: {config.lr}\n"
    model_card += f"- Max Instances: {config.max_instances}\n"

    with open(f"{config.project_name}/README.md", "w") as f:
        f.write(model_card)


def collate_fn(batch):
    """
    Custom collate function for instance segmentation batches.
    
    Args:
        batch (list): List of samples from the dataset
        
    Returns:
        dict: Batched data ready for the model
    """
    pixel_values = []
    labels = []
    
    for item in batch:
        pixel_values.append(item["pixel_values"])
        if "labels" in item:
            labels.append(item["labels"])
    
    # Stack pixel values
    pixel_values = torch.stack(pixel_values)
    
    result = {"pixel_values": pixel_values}
    
    if labels:
        result["labels"] = labels
    
    return result 