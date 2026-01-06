import librosa
import numpy as np
import torch


class AudioSegmentationDataset:
    """
    A dataset class for audio segmentation tasks.
    
    Similar to ImageClassificationDataset but for audio frames.
    Each audio frame gets assigned a label (e.g., speech, music, silence, speaker_1, etc.).

    Args:
        data (Dataset): The dataset containing the audio and frame-level tags.
        processor (PreTrainedProcessor): The processor for audio processing.
        config (Config): Configuration object containing necessary parameters.

    Attributes:
        data (Dataset): The dataset containing the audio and frame-level tags.
        processor (PreTrainedProcessor): The processor for audio processing.
        config (Config): Configuration object containing necessary parameters.

    Methods:
        __len__():
            Returns the number of samples in the dataset.

        __getitem__(item):
            Retrieves a processed audio sample and its corresponding frame-level labels.

            Args:
                item (int): The index of the sample to retrieve.

            Returns:
                dict: A dictionary containing processed audio features and corresponding labels.
    """

    def __init__(self, data, processor, config):
        self.data = data
        self.processor = processor
        self.config = config

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        audio_path = self.data[item][self.config.audio_column]
        tags = self.data[item][self.config.tags_column]

        try:
            audio, _ = librosa.load(audio_path, sr=self.config.sampling_rate)
            
            if len(audio) > self.config.max_length:
                audio = audio[:self.config.max_length]
            
            if len(audio) < self.config.max_length:
                padding = self.config.max_length - len(audio)
                audio = np.pad(audio, (0, padding), mode='constant', constant_values=0)
                
        except Exception as e:
            audio = np.zeros(self.config.max_length)
            
        if self.processor is not None:
            processed_audio = self.processor(
                audio,
                sampling_rate=self.config.sampling_rate,
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt"
            )
            input_values = processed_audio.input_values.squeeze()
            attention_mask = getattr(processed_audio, 'attention_mask', None)
            if attention_mask is not None:
                attention_mask = attention_mask.squeeze()
        else:
            input_values = torch.tensor(audio, dtype=torch.float32)
            attention_mask = None


        frame_reduction = 320
        expected_frames = self.config.max_length // frame_reduction
        
        if isinstance(tags, list) and len(tags) > 0:
            label_ids = tags[:expected_frames]
            
            while len(label_ids) < expected_frames:
                label_ids.append(-100)
        else:
            label_ids = [0] * expected_frames
            
        labels = torch.tensor(label_ids, dtype=torch.long)

        result = {
            "input_values": input_values,
            "labels": labels
        }
        
        if attention_mask is not None:
            result["attention_mask"] = attention_mask

        return result 