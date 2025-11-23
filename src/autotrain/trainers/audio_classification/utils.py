import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn import metrics

from autotrain.trainers.audio_classification.dataset import AudioClassificationDataset


# Constants
BINARY_CLASSIFICATION_EVAL_METRICS = (
    "eval_loss",
    "eval_accuracy",
    "eval_f1",
    "eval_auc",
    "eval_precision",
    "eval_recall",
)

MULTI_CLASS_CLASSIFICATION_EVAL_METRICS = (
    "eval_loss",
    "eval_accuracy",
    "eval_f1_macro",
    "eval_f1_micro",
    "eval_f1_weighted",
    "eval_precision_macro",
    "eval_precision_micro",
    "eval_precision_weighted",
    "eval_recall_macro",
    "eval_recall_micro",
    "eval_recall_weighted",
)

MODEL_CARD = """
---
tags:
- autotrain
- transformers
- audio-classification{base_model}
widget:
- src: https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac
  example_title: Audio sample{dataset_tag}
---

# Model Trained Using AutoTrain

- Problem type: Audio Classification

## Validation Metrics
{validation_metrics}

## Usage

```python
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
import torch
import librosa

# Load model and feature extractor
model = AutoModelForAudioClassification.from_pretrained("YOUR_MODEL_NAME")
feature_extractor = AutoFeatureExtractor.from_pretrained("YOUR_MODEL_NAME")

# Load and preprocess audio
audio, sr = librosa.load("path_to_audio.wav", sr=16000)
inputs = feature_extractor(audio, sampling_rate=sr, return_tensors="pt")

# Make prediction
with torch.no_grad():
    outputs = model(**inputs)
    predicted_class_id = outputs.logits.argmax().item()
    predicted_label = model.config.id2label[predicted_class_id]

print(f"Predicted class: {{predicted_label}}")
```
"""


def _binary_classification_metrics(pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
    """
    Calculate various binary classification metrics for audio classification.

    Args:
        pred: A tuple containing raw predictions and true labels.
              - raw_predictions (numpy.ndarray): The raw prediction scores from the model.
              - labels (numpy.ndarray): The true labels.

    Returns:
        A dictionary containing the following metrics:
        - "f1" (float): The F1 score.
        - "precision" (float): The precision score.
        - "recall" (float): The recall score.
        - "auc" (float): The Area Under the ROC Curve (AUC) score.
        - "accuracy" (float): The accuracy score.
    """
    raw_predictions, labels = pred
    predictions = np.argmax(raw_predictions, axis=1)
    result = {
        "f1": metrics.f1_score(labels, predictions, zero_division=0),
        "precision": metrics.precision_score(labels, predictions, zero_division=0),
        "recall": metrics.recall_score(labels, predictions, zero_division=0),
        "accuracy": metrics.accuracy_score(labels, predictions),
    }
    try:
        result["auc"] = metrics.roc_auc_score(labels, raw_predictions[:, 1])
    except (ValueError, IndexError):
        result["auc"] = 0.0
    return result


def _multi_class_classification_metrics(pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
    """
    Calculate various multi-class classification metrics for audio classification.

    Args:
        pred: A tuple containing raw predictions and true labels.
              - raw_predictions (numpy.ndarray): The raw prediction scores from the model.
              - labels (numpy.ndarray): The true labels.

    Returns:
        A dictionary containing the following metrics:
        - "f1_macro" (float): The macro F1 score.
        - "f1_micro" (float): The micro F1 score.
        - "f1_weighted" (float): The weighted F1 score.
        - "precision_macro" (float): The macro precision score.
        - "precision_micro" (float): The micro precision score.
        - "precision_weighted" (float): The weighted precision score.
        - "recall_macro" (float): The macro recall score.
        - "recall_micro" (float): The micro recall score.
        - "recall_weighted" (float): The weighted recall score.
        - "accuracy" (float): The accuracy score.
    """
    raw_predictions, labels = pred
    predictions = np.argmax(raw_predictions, axis=1)
    return {
        "f1_macro": metrics.f1_score(labels, predictions, average="macro", zero_division=0),
        "f1_micro": metrics.f1_score(labels, predictions, average="micro", zero_division=0),
        "f1_weighted": metrics.f1_score(labels, predictions, average="weighted", zero_division=0),
        "precision_macro": metrics.precision_score(labels, predictions, average="macro", zero_division=0),
        "precision_micro": metrics.precision_score(labels, predictions, average="micro", zero_division=0),
        "precision_weighted": metrics.precision_score(labels, predictions, average="weighted", zero_division=0),
        "recall_macro": metrics.recall_score(labels, predictions, average="macro", zero_division=0),
        "recall_micro": metrics.recall_score(labels, predictions, average="micro", zero_division=0),
        "recall_weighted": metrics.recall_score(labels, predictions, average="weighted", zero_division=0),
        "accuracy": metrics.accuracy_score(labels, predictions),
    }


def process_data(
    train_data: Any,
    valid_data: Optional[Any],
    feature_extractor: Any,
    config: Any,
) -> Tuple[AudioClassificationDataset, Optional[AudioClassificationDataset]]:
    """
    Process training and validation data for audio classification.

    Args:
        train_data: The training dataset.
        valid_data: The validation dataset. Can be None if no validation data is provided.
        feature_extractor: An audio feature extractor.
        config: Configuration dictionary containing additional parameters for dataset processing.

    Returns:
        A tuple containing the processed training dataset and the processed validation dataset 
        (or None if no validation data is provided).
    """
    train_data = AudioClassificationDataset(train_data, feature_extractor, config)
    if valid_data is not None:
        valid_data = AudioClassificationDataset(valid_data, feature_extractor, config)
        return train_data, valid_data
    return train_data, None


def create_model_card(config: Any, trainer: Any, num_classes: int) -> str:
    """
    Generate a model card for the given audio classification configuration and trainer.

    Args:
        config: Configuration object containing various settings.
        trainer: Trainer object used for model training and evaluation.
        num_classes: Number of classes in the classification task.

    Returns:
        A formatted string representing the model card.

    The function evaluates the model if a validation split is provided in the config.
    It then formats the evaluation scores based on whether the task is binary or multi-class classification.
    If no validation split is provided, it notes that no validation metrics are available.

    The function also checks the data path and model path in the config to determine if they are directories.
    Based on these checks, it formats the dataset tag and base model information accordingly.

    Finally, it uses the formatted information to create and return the model card string.
    """
    if config.valid_split is not None:
        eval_scores = trainer.evaluate()
        valid_metrics = (
            BINARY_CLASSIFICATION_EVAL_METRICS if num_classes == 2 else MULTI_CLASS_CLASSIFICATION_EVAL_METRICS
        )
        eval_scores = [f"{k[len('eval_'):]}: {v}" for k, v in eval_scores.items() if k in valid_metrics]
        eval_scores = "\n\n".join(eval_scores)
    else:
        eval_scores = "No validation metrics available"

    if config.data_path == f"{config.project_name}/autotrain-data" or os.path.isdir(config.data_path):
        dataset_tag = ""
    else:
        dataset_tag = f"\ndatasets:\n- {config.data_path}"

    if os.path.isdir(config.model):
        base_model = ""
    else:
        base_model = f"\nbase_model: {config.model}"

    model_card = MODEL_CARD.format(
        dataset_tag=dataset_tag,
        validation_metrics=eval_scores,
        base_model=base_model,
    )
    return model_card 