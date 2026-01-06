import json
import librosa
import numpy as np
import torch


class AudioDetectionDataset:
    """
    A dataset class for audio detection tasks.
    
    Similar to object detection in images, but for temporal events in audio.
    Each audio file can contain multiple events with start/end times and labels.

    Args:
        data (list): A list of data entries where each entry is a dictionary containing audio and events information.
        processor (callable): Audio processor for preprocessing.
        config (object): A configuration object containing column names and audio parameters.

    Attributes:
        data (list): The dataset containing audio and events information.
        processor (callable): The processor for audio preprocessing.
        config (object): The configuration object with column names and parameters.

    Methods:
        __len__(): Returns the number of items in the dataset.
        __getitem__(item): Retrieves and processes the audio and events for the given index.

    Expected data format:
        audio_column: path to audio file or audio array
        events_column: list of events with format:
            [{"start": 4.23, "end": 4.27, "label": "car_crash"}, ...]
    """

    def __init__(self, data, processor, config, label2id=None):
        self.data = data
        self.processor = processor
        self.config = config
        self.label2id = label2id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        audio_data = self.data[item][self.config.audio_column]
        events_data = self.data[item][self.config.events_column]
        
        if isinstance(audio_data, str):
            audio_array, sr = librosa.load(audio_data, sr=self.config.sampling_rate, mono=True)
        elif isinstance(audio_data, dict) and 'array' in audio_data:
            audio_array = audio_data['array']
            sr = audio_data.get('sampling_rate', self.config.sampling_rate)
        elif isinstance(audio_data, np.ndarray):
            audio_array = audio_data
            sr = self.config.sampling_rate
        else:
            raise ValueError(f"Unsupported audio data format: {type(audio_data)}")
            
        if sr != self.config.sampling_rate:
            audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=self.config.sampling_rate)
            
        audio_duration = len(audio_array) / self.config.sampling_rate
        
        if isinstance(events_data, str):
            events = json.loads(events_data)
        elif isinstance(events_data, list):
            events = events_data
        else:
            raise ValueError(f"Unsupported events format: {type(events_data)}")
            
        if len(audio_array) > self.config.max_length:
            audio_array = audio_array[:self.config.max_length]
            audio_duration = self.config.max_length / self.config.sampling_rate
        elif len(audio_array) < self.config.max_length:
            padding = self.config.max_length - len(audio_array)
            audio_array = np.pad(audio_array, (0, padding), mode='constant', constant_values=0)
            
        if self.processor is not None:
            inputs = self.processor(
                audio_array,
                sampling_rate=self.config.sampling_rate,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_length
            )
            audio_values = inputs.input_values.squeeze(0)
            attention_mask = getattr(inputs, 'attention_mask', None)
            if attention_mask is not None:
                attention_mask = attention_mask.squeeze(0)
        else:
            audio_values = torch.tensor(audio_array, dtype=torch.float32)
            attention_mask = torch.ones(len(audio_array), dtype=torch.long)
            
        label_counts = {}
        valid_events = []
        
        for event in events:
            start_time = float(event['start'])
            end_time = float(event['end'])
            label = event['label']
            
            if start_time < audio_duration and end_time <= audio_duration and start_time < end_time:
                valid_events.append(event)
                duration = end_time - start_time
                if label in label_counts:
                    label_counts[label] += duration
                else:
                    label_counts[label] = duration
        
        if label_counts:
            primary_label = max(label_counts.keys(), key=lambda k: label_counts[k])
        else:
            primary_label = events[0]['label'] if events else 'unknown'
        
        if self.label2id and primary_label in self.label2id:
            label_id = self.label2id[primary_label]
        else:
            label_id = 0
                
        result = {
            "input_values": audio_values,
            "labels": torch.tensor(label_id, dtype=torch.long),
            "audio_id": str(item),
            "audio_duration": audio_duration,
            "events": valid_events 
        }
        
        if attention_mask is not None:
            result["attention_mask"] = attention_mask
            
        return result 