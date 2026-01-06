import torch
import librosa
import numpy as np
from typing import Dict, Any


class AudioClassificationDataset:
    """
    A custom dataset class for audio classification tasks.

    Args:
        data (list): A list of data samples, where each sample is a dictionary containing audio and target information.
        processor (callable): A processor that processes audio data.
        config (object): A configuration object containing the column names for audio and targets.

    Attributes:
        data (list): The dataset containing audio and target information.
        processor (callable): The processor to be applied to the audio.
        config (object): The configuration object with audio and target column names.

    Methods:
        __len__(): Returns the number of samples in the dataset.
        __getitem__(item): Retrieves the audio and target at the specified index, processes audio, and returns them as tensors.

    Example:
        dataset = AudioClassificationDataset(data, processor, config)
        audio_features, target = dataset[0]
    """

    def __init__(self, data, processor, config):
        self.data = data
        self.processor = processor
        self.config = config

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item) -> Dict[str, Any]:
        """
        Get a single item from the dataset.
        
        Args:
            item (int): Index of the item to retrieve
            
        Returns:
            Dict[str, Any]: Dictionary containing processed audio features and labels
        """
        audio_data = self.data[item][self.config.audio_column]
        target = int(self.data[item][self.config.target_column])

        # Handle different audio input formats
        if isinstance(audio_data, dict):
            # HuggingFace dataset format
            if 'array' in audio_data and 'sampling_rate' in audio_data:
                audio_array = audio_data['array']
                sampling_rate = audio_data['sampling_rate']
            else:
                raise ValueError("Audio data must contain 'array' and 'sampling_rate' keys")
        elif isinstance(audio_data, str):
            # File path - load the audio file
            audio_array, sampling_rate = librosa.load(
                audio_data, 
                sr=self.config.sampling_rate, 
                mono=True
            )
        elif isinstance(audio_data, np.ndarray):
            # Raw numpy array
            audio_array = audio_data
            sampling_rate = self.config.sampling_rate
        else:
            raise ValueError(f"Unsupported audio data format: {type(audio_data)}")

        # Resample if necessary
        if sampling_rate != self.config.sampling_rate:
            audio_array = librosa.resample(
                audio_array, 
                orig_sr=sampling_rate, 
                target_sr=self.config.sampling_rate
            )

        # Truncate or pad audio to max_length if specified
        if self.config.max_length is not None:
            if len(audio_array) > self.config.max_length:
                audio_array = audio_array[:self.config.max_length]
            elif len(audio_array) < self.config.max_length:
                # Pad with zeros
                padding = self.config.max_length - len(audio_array)
                audio_array = np.pad(audio_array, (0, padding), mode='constant')

        # Process with feature extractor
        try:
            if self.processor is not None:
                processed_audio = self.processor(
                    audio_array,
                    sampling_rate=self.config.sampling_rate,
                    padding=True,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_length
                )
                
                # Extract the first element if batch dimension was added
                processed_inputs = {}
                for key, value in processed_audio.items():
                    if isinstance(value, torch.Tensor) and value.dim() > 1:
                        processed_inputs[key] = value.squeeze(0)
                    else:
                        processed_inputs[key] = value
                        
            processed_inputs = {}
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor) and value.dim() > 1:
                    processed_inputs[key] = value.squeeze(0)
                else:
                    processed_inputs[key] = value
                    
        except Exception as e:
            # Fallback: create basic input structure
            processed_inputs = {
                "input_values": torch.tensor(audio_array, dtype=torch.float32),
            }
            if self.config.feature_extractor_return_attention_mask:
                processed_inputs["attention_mask"] = torch.ones(len(audio_array), dtype=torch.long)

        # Add labels
        processed_inputs["labels"] = torch.tensor(target, dtype=torch.long)

        return processed_inputs 