import argparse
import json

import torch
from accelerate.state import PartialState
from datasets import load_dataset, load_from_disk
from huggingface_hub import HfApi
from transformers import (
    AutoConfig,
    AutoProcessor,
    AutoModelForAudioClassification,
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
from autotrain.trainers.audio_classification import utils
from autotrain.trainers.audio_classification.params import AudioClassificationParams


def parse_args():
    # get training_config.json from the end user
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_config", type=str, required=True)
    return parser.parse_args()


@monitor
def train(config):
    if isinstance(config, dict):
        config = AudioClassificationParams(**config)

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

    # Get classes from the dataset
    if hasattr(train_data.features[config.target_column], 'names'):
        classes = train_data.features[config.target_column].names
    else:
        # If no class names, get unique values
        unique_labels = train_data.unique(config.target_column)
        classes = [f"class_{i}" for i in range(len(unique_labels))]
    
    logger.info(f"Classes: {classes}")
    label2id = {c: i for i, c in enumerate(classes)}
    id2label = {i: c for i, c in enumerate(classes)}
    num_classes = len(classes)

    if num_classes < 2:
        raise ValueError("Invalid number of classes. Must be greater than 1.")

    if config.valid_split is not None:
        if hasattr(valid_data.features[config.target_column], 'names'):
            num_classes_valid = len(valid_data.features[config.target_column].names)
        else:
            num_classes_valid = len(valid_data.unique(config.target_column))
        
        if num_classes_valid != num_classes:
            raise ValueError(
                f"Number of classes in train and valid are not the same. Training has {num_classes} and valid has {num_classes_valid}"
            )

    # Load model configuration
    model_config = AutoConfig.from_pretrained(
        config.model,
        num_labels=num_classes,
        trust_remote_code=ALLOW_REMOTE_CODE,
        token=config.token,
    )
    model_config._num_labels = len(label2id)
    model_config.label2id = label2id
    model_config.id2label = id2label

    # Load model
    try:
        model = AutoModelForAudioClassification.from_pretrained(
            config.model,
            config=model_config,
            trust_remote_code=ALLOW_REMOTE_CODE,
            token=config.token,
            ignore_mismatched_sizes=True,
        )
    except OSError:
        try:
            # Try loading from tf if pytorch version fails
            model = AutoModelForAudioClassification.from_pretrained(
                config.model,
                config=model_config,
                from_tf=True,
                trust_remote_code=ALLOW_REMOTE_CODE,
                token=config.token,
                ignore_mismatched_sizes=True,
            )
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    # Load processor
    try:
        processor = AutoProcessor.from_pretrained(
            config.model,
            token=config.token,
            trust_remote_code=ALLOW_REMOTE_CODE,
        )
    except Exception as e:
        logger.warning(f"Could not load processor: {e}")
        logger.warning("Using default processor settings")
        processor = None

    # Process data
    train_data, valid_data = utils.process_data(train_data, valid_data, processor, config)

    # Set up logging
    if config.logging_steps == -1:
        if config.valid_split is not None:
            logging_steps = int(0.2 * len(train_data) / config.batch_size)
        else:
            logging_steps = int(0.2 * len(train_data) / config.batch_size)
        if logging_steps == 0:
            logging_steps = 1
        if logging_steps > 100:
            logging_steps = 100
        config.logging_steps = logging_steps
    else:
        logging_steps = config.logging_steps

    logger.info(f"Logging steps: {logging_steps}")

    # Training arguments
    training_args = dict(
        output_dir=config.project_name,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=2 * config.batch_size,
        learning_rate=config.lr,
        num_train_epochs=config.epochs,
        eval_strategy=config.eval_strategy if config.valid_split is not None else "no",
        logging_steps=logging_steps,
        save_total_limit=config.save_total_limit,
        save_strategy=config.eval_strategy if config.valid_split is not None else "no",
        gradient_accumulation_steps=config.gradient_accumulation,
        report_to=config.log,
        auto_find_batch_size=config.auto_find_batch_size,
        lr_scheduler_type=config.scheduler,
        optim=config.optimizer,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        push_to_hub=False,
        load_best_model_at_end=True if config.valid_split is not None else False,
        ddp_find_unused_parameters=False,
    )

    if config.mixed_precision == "fp16":
        training_args["fp16"] = True
    if config.mixed_precision == "bf16":
        training_args["bf16"] = True

    # Set up callbacks
    if config.valid_split is not None:
        early_stop = EarlyStoppingCallback(
            early_stopping_patience=config.early_stopping_patience,
            early_stopping_threshold=config.early_stopping_threshold,
        )
        callbacks_to_use = [early_stop]
    else:
        callbacks_to_use = []

    callbacks_to_use.extend([UploadLogs(config=config), LossLoggingCallback(), TrainStartCallback()])

    args = TrainingArguments(**training_args)
    
    # Choose metrics function based on number of classes
    compute_metrics = (
        utils._binary_classification_metrics if num_classes == 2 else utils._multi_class_classification_metrics
    )
    
    trainer_args = dict(
        args=args,
        model=model,
        callbacks=callbacks_to_use,
        compute_metrics=compute_metrics,
    )

    trainer = Trainer(
        **trainer_args,
        train_dataset=train_data,
        eval_dataset=valid_data,
    )
    trainer.remove_callback(PrinterCallback)
    trainer.train()

    logger.info("Finished training, saving model...")
    trainer.save_model(config.project_name)
    
    # Save processor if available
    if processor is not None:
        processor.save_pretrained(config.project_name)

    # Create and save model card
    model_card = utils.create_model_card(config, trainer, num_classes)
    with open(f"{config.project_name}/README.md", "w") as f:
        f.write(model_card)

    # Push to hub if requested
    if config.push_to_hub:
        if PartialState().process_index == 0:
            remove_autotrain_data(config)
            save_training_params(config)
            logger.info("Pushing model to hub...")
            api = HfApi(token=config.token)
            api.create_repo(
                repo_id=f"{config.username}/{config.project_name}", repo_type="model", private=True, exist_ok=True
            )
            api.upload_folder(
                folder_path=config.project_name, repo_id=f"{config.username}/{config.project_name}", repo_type="model"
            )

    if PartialState().process_index == 0:
        pause_space(config)


if __name__ == "__main__":
    _args = parse_args()
    training_config = json.load(open(_args.training_config))
    _config = AudioClassificationParams(**training_config)
    train(_config) 