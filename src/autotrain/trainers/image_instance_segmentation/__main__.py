import argparse
import json
import os

import torch
from accelerate.state import PartialState
from datasets import load_dataset, load_from_disk
from huggingface_hub import HfApi
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_callback import PrinterCallback

from autotrain import logger
from autotrain.trainers.common import (
    ALLOW_REMOTE_CODE,
    LossLoggingCallback,
    TrainStartCallback,
    UploadLogs,
    monitor,
    pause_space,
    remove_autotrain_data,
    save_training_params,
)
from autotrain.trainers.image_instance_segmentation import utils
from autotrain.trainers.image_instance_segmentation.params import ImageInstanceSegmentationParams


def parse_args():
    # get training_config.json from the end user
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_config", type=str, required=True)
    return parser.parse_args()


@monitor
def train(config):
    if isinstance(config, dict):
        config = ImageInstanceSegmentationParams(**config)

    if torch.backends.mps.is_available() and config.mixed_precision in ["fp16", "bf16"]:
        logger.warning(f"{config.mixed_precision} mixed precision is not supported on Apple Silicon MPS. Disabling mixed precision.")
        config.mixed_precision = None

    valid_data = None
    if config.data_path == f"{config.project_name}/autotrain-data":
        train_data = load_from_disk(config.data_path)[config.train_split]
    else:
        if ":" in config.train_split:
            dataset_config_name, split = config.train_split.split(":")
            train_data = load_dataset(
                config.data_path,
                name=dataset_config_name,
                split=split,
                token=config.token,
                trust_remote_code=ALLOW_REMOTE_CODE,
            )
        else:
            train_data = load_dataset(
                config.data_path,
                split=config.train_split,
                token=config.token,
                trust_remote_code=ALLOW_REMOTE_CODE,
            )

    if config.valid_split is not None:
        if config.data_path == f"{config.project_name}/autotrain-data":
            valid_data = load_from_disk(config.data_path)[config.valid_split]
        else:
            if ":" in config.valid_split:
                dataset_config_name, split = config.valid_split.split(":")
                valid_data = load_dataset(
                    config.data_path,
                    name=dataset_config_name,
                    split=split,
                    token=config.token,
                    trust_remote_code=ALLOW_REMOTE_CODE,
                )
            else:
                valid_data = load_dataset(
                    config.data_path,
                    split=config.valid_split,
                    token=config.token,
                    trust_remote_code=ALLOW_REMOTE_CODE,
                )

    logger.info(f"Train data: {train_data}")
    logger.info(f"Valid data: {valid_data}")

    # Load classes from file if it exists
    classes_file = os.path.join(config.data_path, "classes.txt")
    if os.path.exists(classes_file):
        with open(classes_file, "r") as f:
            classes = [line.strip() for line in f.readlines() if line.strip()]
    else:
        # Try to infer classes from the dataset
        logger.warning("No classes.txt file found. Attempting to infer classes from dataset.")
        classes = []
        sample_data = train_data[0] if len(train_data) > 0 else None
        if sample_data and config.category_column in sample_data:
            # This is a simplified approach - in practice, you'd need to scan the entire dataset
            categories = sample_data[config.category_column]
            if isinstance(categories, list):
                unique_categories = set(categories)
                classes = [f"class_{i}" for i in range(max(unique_categories) + 1)]
            else:
                classes = ["class_0", "class_1"]  # Default fallback
        else:
            classes = ["class_0", "class_1"]  # Default fallback
    
    logger.info(f"Classes: {classes}")
    label2id = {c: i for i, c in enumerate(classes)}
    num_classes = len(classes)

    if num_classes < 1:
        raise ValueError("Invalid number of classes. Must be greater than 0.")

    # Configure model
    model_config = AutoConfig.from_pretrained(
        config.model,
        num_labels=num_classes,
        trust_remote_code=ALLOW_REMOTE_CODE,
        token=config.token,
    )
    
    # Set instance segmentation specific configurations
    if hasattr(model_config, 'num_queries'):
        model_config.num_queries = config.max_instances
    
    model_config._num_labels = len(label2id)
    model_config.label2id = label2id
    model_config.id2label = {v: k for k, v in label2id.items()}

    # Load the model - try different model classes based on the model name
    model = None
    model_classes_to_try = [
        "AutoModelForObjectDetection",
        "MaskFormerForInstanceSegmentation", 
        "Mask2FormerForInstanceSegmentation",
        "DEtrForSegmentation",
    ]
    
    for class_name in model_classes_to_try:
        try:
            import transformers
            model_class = getattr(transformers, class_name, None)
            if model_class is None:
                logger.warning(f"Model class {class_name} not found in transformers")
                continue
            
            model = model_class.from_pretrained(
                config.model,
                config=model_config,
                trust_remote_code=ALLOW_REMOTE_CODE,
                token=config.token,
                ignore_mismatched_sizes=True,
            )
            logger.info(f"Successfully loaded model using {class_name}")
            break
        except Exception as e:
            logger.warning(f"Failed to load model with {class_name}: {e}")
            continue
    
    if model is None:
        # Final fallback - try generic AutoModel
        try:
            from transformers import AutoModel
            model = AutoModel.from_pretrained(
                config.model,
                config=model_config,
                trust_remote_code=ALLOW_REMOTE_CODE,
                token=config.token,
                ignore_mismatched_sizes=True,
            )
            logger.info("Loaded model using AutoModel as fallback")
        except Exception as e:
            raise RuntimeError(f"Failed to load any compatible model for {config.model}: {e}")

    # Load image processor
    image_processor = AutoImageProcessor.from_pretrained(
        config.model,
        token=config.token,
        trust_remote_code=ALLOW_REMOTE_CODE,
    )

    # Process data
    train_data, valid_data = utils.process_data(train_data, valid_data, image_processor, config)

    if config.logging_steps == -1:
        if config.valid_split is not None:
            logging_steps = int(0.2 * len(valid_data) / config.batch_size)
        else:
            logging_steps = int(0.2 * len(train_data) / config.batch_size)
        if logging_steps == 0:
            logging_steps = 1
        if logging_steps > 25:
            logging_steps = 25
        config.logging_steps = logging_steps

    logger.info(f"Logging steps: {config.logging_steps}")

    training_args_dict = {
        "output_dir": config.project_name,
        "per_device_train_batch_size": config.batch_size,
        "per_device_eval_batch_size": config.batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation,
        "learning_rate": config.lr,
        "num_train_epochs": config.epochs,
        "eval_strategy": config.eval_strategy if config.valid_split is not None else "no",
        "logging_steps": config.logging_steps,
        "save_strategy": config.save_strategy if config.save_strategy else (config.eval_strategy if config.valid_split is not None else "epoch"),
        "save_total_limit": config.save_total_limit,
        "load_best_model_at_end": True if config.eval_strategy != "no" and config.valid_split is not None else False,
        "warmup_ratio": config.warmup_ratio,
        "weight_decay": config.weight_decay,
        "optim": config.optimizer,
        "lr_scheduler_type": config.scheduler,
        "push_to_hub": config.push_to_hub,
        "report_to": config.log,
        "seed": config.seed,
        "remove_unused_columns": False,
        "dataloader_pin_memory": False,
    }
    
    # Only add hub-related parameters if pushing to hub
    if config.push_to_hub:
        training_args_dict["hub_strategy"] = "every_save"
        training_args_dict["hub_model_id"] = config.hub_model_id
        training_args_dict["hub_token"] = config.token
        training_args_dict["hub_private_repo"] = True
    
    training_args = TrainingArguments(**training_args_dict)

    callbacks = []
    callbacks.append(TrainStartCallback())
    callbacks.append(UploadLogs(config=config))
    callbacks.append(LossLoggingCallback())
    if config.early_stopping_patience > 0 and config.valid_split is not None:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=valid_data,
        tokenizer=image_processor,
        compute_metrics=utils.compute_metrics,
        callbacks=callbacks,
        data_collator=utils.collate_fn,
    )

    trainer.remove_callback(PrinterCallback)
    trainer.train()

    utils.create_model_card(config, trainer, num_classes)

    trainer.save_model()
    if config.push_to_hub:
        trainer.push_to_hub()

    if not config.push_to_hub:
        save_training_params(config)

    remove_autotrain_data(config)
    pause_space(config)


if __name__ == "__main__":
    _args = parse_args()
    training_config = json.load(open(_args.training_config))
    _config = ImageInstanceSegmentationParams(**training_config)
    train(_config) 