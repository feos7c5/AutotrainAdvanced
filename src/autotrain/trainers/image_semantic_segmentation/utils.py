import os

import albumentations as A
import numpy as np
from sklearn import metrics

from autotrain.trainers.image_semantic_segmentation.dataset import ImageSemanticSegmentationDataset


SEGMENTATION_EVAL_METRICS = (
    "eval_loss",
    "eval_accuracy",
    "eval_mean_iou",
)

MODEL_CARD = """
---
tags:
- autotrain
- transformers
- image-segmentation{base_model}
widget:
- src: https://huggingface.co/datasets/mishig/sample_images/resolve/main/tiger.jpg
  example_title: Tiger
- src: https://huggingface.co/datasets/mishig/sample_images/resolve/main/teapot.jpg
  example_title: Teapot
- src: https://huggingface.co/datasets/mishig/sample_images/resolve/main/palace.jpg
  example_title: Palace{dataset_tag}
---

# Model Trained Using AutoTrain

- Problem type: Image Semantic Segmentation
"""


def binary_classification_metrics(y_true, y_pred):
    return {
        "accuracy": metrics.accuracy_score(y_true, y_pred),
        "f1": metrics.f1_score(y_true, y_pred),
        "precision": metrics.precision_score(y_true, y_pred),
        "recall": metrics.recall_score(y_true, y_pred),
        "auc": metrics.roc_auc_score(y_true, y_pred),
    }


def multi_class_classification_metrics(y_true, y_pred):
    return {
        "accuracy": metrics.accuracy_score(y_true, y_pred),
        "f1_macro": metrics.f1_score(y_true, y_pred, average="macro"),
        "f1_micro": metrics.f1_score(y_true, y_pred, average="micro"),
        "f1_weighted": metrics.f1_score(y_true, y_pred, average="weighted"),
        "precision_macro": metrics.precision_score(y_true, y_pred, average="macro"),
        "precision_micro": metrics.precision_score(y_true, y_pred, average="micro"),
        "precision_weighted": metrics.precision_score(y_true, y_pred, average="weighted"),
        "recall_macro": metrics.recall_score(y_true, y_pred, average="macro"),
        "recall_micro": metrics.recall_score(y_true, y_pred, average="micro"),
        "recall_weighted": metrics.recall_score(y_true, y_pred, average="weighted"),
    }


def compute_metrics(eval_pred):
    """Compute metrics for evaluation."""
    predictions, labels = eval_pred
    
    # Get predicted classes by taking argmax along the class dimension
    predictions = np.argmax(predictions, axis=1)
    
    # Handle size mismatch: upsample predictions to match labels
    if predictions.shape != labels.shape:
        # Use scipy for upsampling
        from scipy.ndimage import zoom
        
        # Calculate zoom factors for height and width (batch dimension stays the same)
        zoom_h = labels.shape[1] / predictions.shape[1]
        zoom_w = labels.shape[2] / predictions.shape[2]
        
        # Upsample each prediction in the batch
        upsampled_predictions = []
        for i in range(predictions.shape[0]):
            # Use nearest neighbor interpolation for discrete class labels
            upsampled_pred = zoom(predictions[i], (zoom_h, zoom_w), order=0)
            upsampled_predictions.append(upsampled_pred)
        
        predictions = np.array(upsampled_predictions)
    
    # Flatten the predictions and labels
    predictions = predictions.flatten()
    labels = labels.flatten()
    
    # Remove ignore index (-100)
    valid_mask = labels != -100
    predictions = predictions[valid_mask]
    labels = labels[valid_mask]
    
    # Calculate accuracy
    if len(predictions) == 0 or len(labels) == 0:
        return {"accuracy": 0.0}
    
    accuracy = metrics.accuracy_score(labels, predictions)
    
    return {"accuracy": accuracy}


def process_data(train_data, valid_data, image_processor, config):
    """
    Processes training and validation data for image semantic segmentation.

    Args:
        train_data (Dataset): The training dataset.
        valid_data (Dataset or None): The validation dataset. Can be None if no validation data is provided.
        image_processor (ImageProcessor): An object containing image processing parameters such as size, mean, and std.
        config (dict): Configuration dictionary containing additional parameters for dataset processing.

    Returns:
        tuple: A tuple containing the processed training dataset and the processed validation dataset (or None if no validation data is provided).
    """
    if "shortest_edge" in image_processor.size:
        size = image_processor.size["shortest_edge"]
    else:
        size = (image_processor.size["height"], image_processor.size["width"])
    try:
        height, width = size
    except TypeError:
        height = size
        width = size

    train_transforms = A.Compose(
        [
            A.Resize(height=height, width=width),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=image_processor.image_mean, std=image_processor.image_std),
        ],
        is_check_shapes=False
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=height, width=width),
            A.Normalize(mean=image_processor.image_mean, std=image_processor.image_std),
        ],
        is_check_shapes=False
    )
    train_data = ImageSemanticSegmentationDataset(train_data, train_transforms, config)
    if valid_data is not None:
        valid_data = ImageSemanticSegmentationDataset(valid_data, val_transforms, config)
        return train_data, valid_data
    return train_data, None


def create_model_card(config, trainer, num_classes):
    """
    Generates a model card for the given configuration and trainer.

    Args:
        config (object): Configuration object containing various settings.
        trainer (object): Trainer object used for model training and evaluation.
        num_classes (int): Number of classes in the segmentation task.

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
            if "eval_accuracy" in log:
                eval_results += f"- Eval Accuracy: {log['eval_accuracy']:.4f}\n"
            if "train_loss" in log:
                eval_results += f"- Train Loss: {log['train_loss']:.4f}\n"
    
    model_card += f"\n\n{eval_results}"
    model_card += f"\n\n## Model Details\n\n"
    model_card += f"- Problem Type: Image Semantic Segmentation\n"
    model_card += f"- Model Architecture: {config.model}\n"
    model_card += f"- Number of Classes: {num_classes}\n"
    model_card += f"- Training Epochs: {config.epochs}\n"
    model_card += f"- Batch Size: {config.batch_size}\n"
    model_card += f"- Learning Rate: {config.lr}\n"

    with open(f"{config.project_name}/README.md", "w") as f:
        f.write(model_card) 