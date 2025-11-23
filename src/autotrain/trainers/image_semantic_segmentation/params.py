from typing import Optional

from pydantic import Field

from autotrain.trainers.common import AutoTrainParams


class ImageSemanticSegmentationParams(AutoTrainParams):
    """
    ImageSemanticSegmentationParams is a configuration class for image semantic segmentation training parameters.

    Attributes:
        data_path (str): Path to the dataset.
        model (str): Pre-trained model name or path. Default is "facebook/detr-resnet-50-panoptic".
        username (Optional[str]): Hugging Face account username.
        lr (float): Learning rate for the optimizer. Default is 5e-5.
        epochs (int): Number of epochs for training. Default is 3.
        batch_size (int): Batch size for training. Default is 8.
        warmup_ratio (float): Warmup ratio for learning rate scheduler. Default is 0.1.
        gradient_accumulation (int): Number of gradient accumulation steps. Default is 1.
        optimizer (str): Optimizer type. Default is "adamw_torch".
        scheduler (str): Learning rate scheduler type. Default is "linear".
        weight_decay (float): Weight decay for the optimizer. Default is 0.0.
        max_grad_norm (float): Maximum gradient norm for clipping. Default is 1.0.
        seed (int): Random seed for reproducibility. Default is 42.
        train_split (str): Name of the training data split. Default is "train".
        valid_split (Optional[str]): Name of the validation data split.
        logging_steps (int): Number of steps between logging. Default is -1.
        project_name (str): Name of the project for output directory. Default is "project-name".
        auto_find_batch_size (bool): Automatically find optimal batch size. Default is False.
        mixed_precision (Optional[str]): Mixed precision training mode (fp16, bf16, or None).
        save_total_limit (int): Maximum number of checkpoints to keep. Default is 1.
        token (Optional[str]): Hugging Face Hub token for authentication.
        push_to_hub (bool): Whether to push the model to Hugging Face Hub. Default is False.
        eval_strategy (str): Evaluation strategy during training. Default is "epoch".
        save_strategy (Optional[str]): Save strategy for checkpoints (epoch, steps, no). Default is None.
        image_column (str): Column name for images in the dataset. Default is "image".
        target_column (str): Column name for target segmentation masks in the dataset. Default is "segmentation_mask".
        log (str): Logging method for experiment tracking. Default is "none".
        early_stopping_patience (int): Number of epochs with no improvement for early stopping. Default is 5.
        early_stopping_threshold (float): Threshold for early stopping. Default is 0.01.
        ignore_mismatched_sizes (bool): Whether to ignore mismatched sizes when loading model. Default is True.
        reduce_labels (bool): Whether to reduce label ids by 1 for datasets where 0 is background. Default is False.
        
    """
    
    data_path: str = Field(None, title="Data path")
    model: str = Field("nvidia/mit-b0", title="Model")
    username: Optional[str] = Field(None, title="Hugging Face Username")
    lr: float = Field(5e-5, title="Learning rate")
    epochs: int = Field(3, title="Number of training epochs")
    batch_size: int = Field(2, title="Training batch size")
    warmup_ratio: float = Field(0.1, title="Warmup proportion")
    gradient_accumulation: int = Field(1, title="Gradient accumulation steps")
    optimizer: str = Field("adamw_torch", title="Optimizer")
    scheduler: str = Field("linear", title="Scheduler")
    weight_decay: float = Field(0.0, title="Weight decay")
    max_grad_norm: float = Field(1.0, title="Max gradient norm")
    seed: int = Field(42, title="Seed")
    train_split: str = Field("train", title="Train split")
    valid_split: Optional[str] = Field(None, title="Validation split")
    logging_steps: int = Field(-1, title="Logging steps")
    project_name: str = Field("project-name", title="Output directory")
    auto_find_batch_size: bool = Field(False, title="Auto find batch size")
    mixed_precision: Optional[str] = Field(None, title="fp16, bf16, or None")
    save_total_limit: int = Field(1, title="Save total limit")
    token: Optional[str] = Field(None, title="Hub Token")
    push_to_hub: bool = Field(False, title="Push to hub")
    eval_strategy: str = Field("epoch", title="Evaluation strategy")
    save_strategy: Optional[str] = Field(None, title="Save strategy (epoch, steps, no)")
    image_column: str = Field("image", title="Image column")
    target_column: str = Field("segmentation_mask", title="Target column")
    log: str = Field("none", title="Logging using experiment tracking")
    early_stopping_patience: int = Field(5, title="Early stopping patience")
    early_stopping_threshold: float = Field(0.01, title="Early stopping threshold")
    ignore_mismatched_sizes: bool = Field(True, title="Ignore mismatched sizes")
    reduce_labels: bool = Field(False, title="Reduce label ids by 1")
    hub_model_id: Optional[str] = Field(None, title="Hub Model ID") 