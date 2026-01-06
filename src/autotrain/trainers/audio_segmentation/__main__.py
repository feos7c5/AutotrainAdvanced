import argparse
import json
import os
from functools import partial

import torch
import torch.nn as nn
from accelerate.state import PartialState
from datasets import load_dataset, load_from_disk
from huggingface_hub import HfApi
from transformers import (
    AutoConfig,
    AutoProcessor,
    AutoModel,
    EarlyStoppingCallback,
    PreTrainedModel,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
)
from transformers.modeling_outputs import SequenceClassifierOutput
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
from autotrain.trainers.audio_segmentation import utils
from autotrain.trainers.audio_segmentation.dataset import AudioSegmentationDataset
from autotrain.trainers.audio_segmentation.params import AudioSegmentationParams


class AudioSegmentationConfig(PretrainedConfig):
    """
    Configuration for Audio Segmentation models.
    Similar to how image_classification handles custom configs.
    """
    model_type = "audio_segmentation"
    
    def __init__(
        self,
        backbone_model_name: str = "facebook/wav2vec2-base",
        num_labels: int = 2,
        hidden_dropout: float = 0.1,
        final_dropout: float = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.backbone_model_name = backbone_model_name
        self.num_labels = num_labels
        self.hidden_dropout = hidden_dropout
        self.final_dropout = final_dropout


class AudioSegmentationModel(PreTrainedModel):
    """
    Generic Audio Segmentation Model for frame-level predictions.
    Similar structure to image_classification but for audio frames.
    """
    config_class = AudioSegmentationConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        backbone_config = AutoConfig.from_pretrained(
            config.backbone_model_name,
            trust_remote_code=ALLOW_REMOTE_CODE
        )
        
        self.backbone = AutoModel.from_pretrained(
            config.backbone_model_name,
            config=backbone_config,
            trust_remote_code=ALLOW_REMOTE_CODE
        )
        
        if hasattr(backbone_config, 'hidden_size'):
            hidden_size = backbone_config.hidden_size
        elif hasattr(backbone_config, 'd_model'):
            hidden_size = backbone_config.d_model
        else:
            hidden_size = 768 
            
        self.dropout = nn.Dropout(config.final_dropout)
        self.classifier = nn.Linear(hidden_size, config.num_labels)
        
        self.post_init()
        
    def forward(
        self,
        input_values=None,
        attention_mask=None,
        labels=None,
        **kwargs
    ):
        outputs = self.backbone(
            input_values,
            attention_mask=attention_mask,
            **kwargs
        )
        
        if hasattr(outputs, 'last_hidden_state'):
            sequence_output = outputs.last_hidden_state
        else:
            sequence_output = outputs[0]
            
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

            seq_len = min(logits.size(1), labels.size(1))
            active_logits = logits[:, :seq_len, :].contiguous().view(-1, self.num_labels)
            active_labels = labels[:, :seq_len].contiguous().view(-1)
            
            loss = loss_fct(active_logits, active_labels)
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
            attentions=outputs.attentions if hasattr(outputs, 'attentions') else None,
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_config", type=str, required=True)
    return parser.parse_args()


@monitor
def train(config):
    if isinstance(config, dict):
        config = AudioSegmentationParams(**config)

    if torch.backends.mps.is_available() and config.mixed_precision in ["fp16", "bf16"]:
        logger.warning(f"{config.mixed_precision} mixed precision is not supported on Apple Silicon MPS. Disabling mixed precision.")
        config.mixed_precision = None

    train_data = None
    valid_data = None
    
    if config.train_split is not None:
        if config.data_path == f"{config.project_name}/autotrain-data":
            logger.info("loading dataset from disk")
            train_data = load_from_disk(config.data_path)[config.train_split]
        else:
            if os.path.isdir(config.data_path) and os.path.exists(os.path.join(config.data_path, "dataset_info.json")):
                logger.info("loading dataset from disk (save_to_disk format)")
                train_data = load_from_disk(config.data_path)
                if hasattr(train_data, 'train'):
                    train_data = train_data['train']
            elif config.data_path.endswith('.csv'):
                train_data = load_dataset('csv', data_files=config.data_path, split='train')
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
            logger.info("loading dataset from disk")
            valid_data = load_from_disk(config.data_path)[config.valid_split]
        else:
            if os.path.isdir(config.data_path) and os.path.exists(os.path.join(config.data_path, "dataset_info.json")):
                logger.info("loading validation dataset from disk (save_to_disk format)")
                loaded_data = load_from_disk(config.data_path)
                if hasattr(loaded_data, config.valid_split):
                    valid_data = loaded_data[config.valid_split]
                else:
                    valid_data = None
            elif config.data_path.endswith('.csv'):
                valid_data = None  
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

    # Handle both ClassLabel and Value (list of integers) features
    if hasattr(train_data.features[config.tags_column], 'feature') and hasattr(train_data.features[config.tags_column].feature, 'names'):
        # ClassLabel feature type
        label_list = train_data.features[config.tags_column].feature.names
    else:
        # Value feature type (list of integers) - extract unique labels
        all_labels = set()
        for example in train_data:
            all_labels.update(example[config.tags_column])
        label_list = sorted(list(all_labels))
        # Convert integer labels to string labels for consistency
        label_list = [f"label_{i}" for i in label_list]
    
    num_classes = len(label_list)

    if num_classes < 2:
        raise ValueError("Invalid number of classes. Must be greater than 1.")

    logger.info(f"Audio segmentation labels: {label_list}")
    logger.info(f"Number of classes: {num_classes}")

    if config.valid_split is not None and valid_data is not None:
        # Same logic for validation data
        if hasattr(valid_data.features[config.tags_column], 'feature') and hasattr(valid_data.features[config.tags_column].feature, 'names'):
            valid_label_list = valid_data.features[config.tags_column].feature.names
        else:
            valid_all_labels = set()
            for example in valid_data:
                valid_all_labels.update(example[config.tags_column])
            valid_label_list = sorted(list(valid_all_labels))
            valid_label_list = [f"label_{i}" for i in valid_label_list]
        
        if len(valid_label_list) != num_classes:
            raise ValueError(
                f"Number of classes in train and valid are not the same. Training has {num_classes} and valid has {len(valid_label_list)}"
            )

    model_config = AudioSegmentationConfig(
        backbone_model_name=config.model,
        num_labels=num_classes,
        hidden_dropout=0.1,
        final_dropout=0.1,
    )
    
    model = AudioSegmentationModel(model_config)

    try:
        processor = AutoProcessor.from_pretrained(
            config.model,
            token=config.token,
            trust_remote_code=ALLOW_REMOTE_CODE,
        )
        
        if hasattr(processor, 'sampling_rate'):
            processor.sampling_rate = config.sampling_rate
            
    except Exception as e:
        logger.warning(f"Could not load processor: {e}")
        processor = None

    train_data = AudioSegmentationDataset(data=train_data, processor=processor, config=config)
    if config.valid_split is not None and valid_data is not None:
        valid_data = AudioSegmentationDataset(data=valid_data, processor=processor, config=config)

    if config.logging_steps == -1:
        if config.valid_split is not None and valid_data is not None:
            logging_steps = int(0.2 * len(valid_data) / config.batch_size)
        else:
            logging_steps = int(0.2 * len(train_data) / config.batch_size)
        if logging_steps == 0:
            logging_steps = 1
        if logging_steps > 25:
            logging_steps = 25
        config.logging_steps = logging_steps
    else:
        logging_steps = config.logging_steps

    logger.info(f"Logging steps: {logging_steps}")
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

    if config.valid_split is not None and valid_data is not None:
        early_stop = EarlyStoppingCallback(
            early_stopping_patience=config.early_stopping_patience,
            early_stopping_threshold=config.early_stopping_threshold,
        )
        callbacks_to_use = [early_stop]
    else:
        callbacks_to_use = []

    callbacks_to_use.extend([UploadLogs(config=config), LossLoggingCallback(), TrainStartCallback()])

    args = TrainingArguments(**training_args)
    trainer_args = dict(
        args=args,
        model=model,
        callbacks=callbacks_to_use,
        compute_metrics=partial(utils.audio_segmentation_metrics, label_list=label_list),
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
    
    if processor is not None:
        processor.save_pretrained(config.project_name)

    model_card = utils.create_model_card(config, trainer)

    with open(f"{config.project_name}/README.md", "w", encoding="utf-8") as f:
        f.write(model_card)
    
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
                folder_path=config.project_name,
                repo_id=f"{config.username}/{config.project_name}",
                repo_type="model",
            )

    if PartialState().process_index == 0:
        pause_space(config)


if __name__ == "__main__":
    args = parse_args()
    training_config = json.load(open(args.training_config))
    config = AudioSegmentationParams(**training_config)
    train(config) 