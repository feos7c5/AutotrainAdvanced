import json
import os

import librosa
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from seqeval import metrics

from autotrain import logger


AUDIO_SEGMENTATION_EVAL_METRICS = (
    "eval_loss",
    "eval_accuracy",
    "eval_f1_macro",
    "eval_f1_micro",
    "eval_precision",
    "eval_recall",
)

MODEL_CARD = """
---
library_name: transformers
tags:
- autotrain
- audio-segmentation{base_model}
widget:
- example_title: "Audio Segmentation Example"
  src: "https://cdn-media.huggingface.co/speech_samples/sample1.flac"{dataset_tag}
---

# Model Trained Using AutoTrain

- Problem type: Audio Segmentation

## Validation Metrics
{validation_metrics}
"""


def process_data(train_data, valid_data, feature_extractor, config):
    """
    Process audio segmentation data for training.
    
    Args:
        train_data (Dataset): Training dataset
        valid_data (Dataset or None): Validation dataset
        feature_extractor (FeatureExtractor): Audio feature extractor
        config (object): Training configuration
        
    Returns:
        tuple: Processed training and validation datasets
    """
    def preprocess_function(examples):
        """Preprocess audio data for segmentation."""
        audios = []
        labels = []
        
        for audio_path, segments in zip(examples[config.audio_column], examples[config.target_column]):
            try:
                audio, sr = librosa.load(audio_path, sr=config.sampling_rate)
                
                if len(audio) > config.max_length:
                    audio = audio[:config.max_length]
                
                if len(audio) < config.max_length:
                    padding = config.max_length - len(audio)
                    audio = np.pad(audio, (0, padding), mode='constant', constant_values=0)
                
                audios.append(audio)
                
                if isinstance(segments, str):
                    segments = json.loads(segments)
                
                frame_labels = create_frame_labels(segments, len(audio), config.sampling_rate)
                labels.append(frame_labels)
                
            except Exception as e:
                logger.warning(f"Error processing audio {audio_path}: {e}")
                audios.append(np.zeros(config.max_length))
                labels.append(np.zeros(config.max_length // 1000))
        
        if feature_extractor is not None:
            inputs = feature_extractor(
                audios,
                sampling_rate=config.sampling_rate,
                padding=True,
                truncation=True,
                max_length=config.max_length,
                return_tensors="pt"
            )
            
            result = {
                "input_values": inputs.input_values.tolist(),
                "labels": labels
            }
            
            if hasattr(inputs, "attention_mask") and inputs.attention_mask is not None:
                result["attention_mask"] = inputs.attention_mask.tolist()
                
            return result
        else:
            return {
                "input_values": audios,
                "labels": labels
            }

    train_data = train_data.map(
        preprocess_function,
        batched=True,
        remove_columns=train_data.column_names,
        desc="Processing training data"
    )
    
    if valid_data is not None:
        valid_data = valid_data.map(
            preprocess_function,
            batched=True,
            remove_columns=valid_data.column_names,
            desc="Processing validation data"
        )
    
    return train_data, valid_data


def create_frame_labels(segments, audio_length, sampling_rate):
    """
    Create frame-level labels from segment annotations.
    
    Args:
        segments (list): List of segment dictionaries with 'start', 'end', 'label'
        audio_length (int): Length of audio in samples
        sampling_rate (int): Audio sampling rate
        
    Returns:
        numpy.ndarray: Frame-level labels array
    """
    frame_rate = 100
    num_frames = int(audio_length / sampling_rate * frame_rate)
    frame_labels = np.zeros(num_frames, dtype=np.int64)
    
    for segment in segments:
        start_frame = int(segment['start'] * frame_rate)
        end_frame = int(segment['end'] * frame_rate)
        label = segment.get('label', 1)
        
        if isinstance(label, str):
            label = hash(label) % 10
        
        start_frame = max(0, start_frame)
        end_frame = min(num_frames, end_frame)
        
        frame_labels[start_frame:end_frame] = label
    
    return frame_labels


def _segmentation_metrics(eval_pred):
    """
    Calculate segmentation metrics.
    
    Args:
        eval_pred (EvalPrediction): EvalPrediction object containing predictions and labels
        
    Returns:
        dict: Dictionary of metrics
    """
    predictions, labels = eval_pred

    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    
    if predictions.ndim > 2:
        predictions = np.argmax(predictions, axis=-1)
    
    predictions_flat = predictions.flatten()
    labels_flat = labels.flatten()
    
    mask = labels_flat != -100
    predictions_flat = predictions_flat[mask]
    labels_flat = labels_flat[mask]
    
    accuracy = accuracy_score(labels_flat, predictions_flat)
    f1_macro = f1_score(labels_flat, predictions_flat, average='macro', zero_division=0)
    f1_micro = f1_score(labels_flat, predictions_flat, average='micro', zero_division=0)
    precision = precision_score(labels_flat, predictions_flat, average='macro', zero_division=0)
    recall = recall_score(labels_flat, predictions_flat, average='macro', zero_division=0)
    
    return {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_micro": f1_micro,
        "precision": precision,
        "recall": recall,
    }


def audio_segmentation_metrics(pred, label_list):
    """
    Compute audio segmentation metrics including precision, recall, F1 score, and accuracy.
    
    Audio segmentation is similar to token classification but for audio frames instead of text tokens.
    Each audio frame gets assigned a label (e.g., speech, music, silence, speaker_1, etc.).

    Args:
        pred (tuple): A tuple containing predictions and labels.
                      Predictions should be a 3D array (batch_size, sequence_length, num_labels).
                      Labels should be a 2D array (batch_size, sequence_length).
        label_list (list): A list of label names corresponding to the indices used in predictions and labels.

    Returns:
        dict: A dictionary containing the following metrics:
              - "precision": Precision score of the audio segmentation.
              - "recall": Recall score of the audio segmentation.
              - "f1": F1 score of the audio segmentation.
              - "accuracy": Accuracy score of the audio segmentation.
    """
    predictions, labels = pred
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[predi] for (predi, lbl) in zip(prediction, label) if lbl != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[lbl] for (predi, lbl) in zip(prediction, label) if lbl != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = {
        "precision": metrics.precision_score(true_labels, true_predictions),
        "recall": metrics.recall_score(true_labels, true_predictions),
        "f1": metrics.f1_score(true_labels, true_predictions),
        "accuracy": metrics.accuracy_score(true_labels, true_predictions),
    }
    return results


def create_model_card(config, trainer):
    """
    Generates a model card string based on the provided configuration and trainer.

    Args:
        config (object): Configuration object containing model and dataset information.
        trainer (object): Trainer object used to evaluate the model.

    Returns:
        str: A formatted model card string with dataset tags, validation metrics, and base model information.
    """
    if config.valid_split is not None:
        eval_scores = trainer.evaluate()
        valid_metrics = ["eval_loss", "eval_precision", "eval_recall", "eval_f1", "eval_accuracy"]
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