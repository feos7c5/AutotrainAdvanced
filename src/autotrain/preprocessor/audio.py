import os
from dataclasses import dataclass
from typing import Optional, Union

import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split

from autotrain import logger


@dataclass
class AudioClassificationPreprocessor:
    """
    A preprocessor for audio classification datasets.

    Attributes:
        train_data (str): Path to the training data file or directory.
        valid_data (Optional[str]): Path to the validation data file or directory (optional).
        test_data (Optional[str]): Path to the test data file or directory (optional).
        token (Optional[str]): Hugging Face Hub token for uploading datasets.
        project_name (str): Name of the project for output directory.
        username (Optional[str]): Hugging Face username for uploading datasets.
        valid_split (float): Proportion of training data to use for validation if no validation data is provided.
        seed (int): Random seed for reproducible train/validation splits.
        local (bool): Whether to save the dataset locally or upload to Hugging Face Hub.
        audio_column (str): Name of the column containing audio file paths or audio data.
        target_column (str): Name of the column containing target labels.

    Methods:
        split(): Splits the data into training and validation sets.
        prepare_columns(train_df, valid_df): Prepares the column names for the datasets.
        prepare(): Main preprocessing method that prepares the complete dataset.
    """

    train_data: Union[str, pd.DataFrame]
    valid_data: Optional[Union[str, pd.DataFrame]] = None
    test_data: Optional[Union[str, pd.DataFrame]] = None
    token: Optional[str] = None
    project_name: str = "project-name"
    username: Optional[str] = None
    valid_split: float = 0.2
    seed: int = 42
    local: bool = True
    audio_column: str = "audio"
    target_column: str = "target"

    def __post_init__(self):
        """Post-initialization validation and setup."""
        # Validate train_data
        if isinstance(self.train_data, str):
            if not os.path.exists(self.train_data):
                raise ValueError(f"Training data path does not exist: {self.train_data}")
        elif not isinstance(self.train_data, pd.DataFrame):
            raise ValueError(f"Training data must be a string path or pandas DataFrame, got: {type(self.train_data)}")
        
        # Validate valid_data if provided
        if self.valid_data:
            if isinstance(self.valid_data, str):
                if not os.path.exists(self.valid_data):
                    raise ValueError(f"Validation data path does not exist: {self.valid_data}")
            elif not isinstance(self.valid_data, pd.DataFrame):
                raise ValueError(f"Validation data must be a string path or pandas DataFrame, got: {type(self.valid_data)}")
        
        # Validate test_data if provided
        if self.test_data:
            if isinstance(self.test_data, str):
                if not os.path.exists(self.test_data):
                    raise ValueError(f"Test data path does not exist: {self.test_data}")
            elif not isinstance(self.test_data, pd.DataFrame):
                raise ValueError(f"Test data must be a string path or pandas DataFrame, got: {type(self.test_data)}")

    def split(self):
        """
        Splits the training data into training and validation sets if no validation data is provided.

        Returns:
            tuple: A tuple containing (train_df, valid_df) DataFrames.
        """
        # Load training data
        if isinstance(self.train_data, pd.DataFrame):
            train_df = self.train_data.copy()
        else:
            # Check if it's a directory with metadata.jsonl (ZIP format)
            if os.path.isdir(self.train_data):
                train_df = self._load_from_metadata_jsonl(self.train_data)
            # Load from file path (CSV/JSONL format)
            elif self.train_data.endswith(('.csv', '.tsv')):
                separator = '\t' if self.train_data.endswith('.tsv') else ','
                train_df = pd.read_csv(self.train_data, sep=separator)
            elif self.train_data.endswith('.jsonl'):
                train_df = pd.read_json(self.train_data, lines=True)
            elif self.train_data.endswith('.json'):
                train_df = pd.read_json(self.train_data)
            else:
                raise ValueError(f"Unsupported file format for training data: {self.train_data}")

        # Load validation data if provided
        if self.valid_data:
            if isinstance(self.valid_data, pd.DataFrame):
                valid_df = self.valid_data.copy()
            else:
                # Check if it's a directory with metadata.jsonl (ZIP format)
                if os.path.isdir(self.valid_data):
                    valid_df = self._load_from_metadata_jsonl(self.valid_data)
                # Load from file path (CSV/JSONL format)
                elif self.valid_data.endswith(('.csv', '.tsv')):
                    separator = '\t' if self.valid_data.endswith('.tsv') else ','
                    valid_df = pd.read_csv(self.valid_data, sep=separator)
                elif self.valid_data.endswith('.jsonl'):
                    valid_df = pd.read_json(self.valid_data, lines=True)
                elif self.valid_data.endswith('.json'):
                    valid_df = pd.read_json(self.valid_data)
                else:
                    raise ValueError(f"Unsupported file format for validation data: {self.valid_data}")
        else:
            # Split training data into train and validation
            train_df, valid_df = train_test_split(
                train_df,
                test_size=self.valid_split,
                random_state=self.seed,
                stratify=train_df[self.target_column] if self.target_column in train_df.columns else None
            )

        logger.info(f"Training data shape: {train_df.shape}")
        logger.info(f"Validation data shape: {valid_df.shape}")

        return train_df, valid_df

    def _load_from_metadata_jsonl(self, data_dir):
        """
        Load data from a directory containing audio files and metadata.jsonl
        
        Args:
            data_dir (str): Directory containing audio files and metadata.jsonl
            
        Returns:
            pd.DataFrame: DataFrame with file_name and label columns
        """
        import json
        
        metadata_path = os.path.join(data_dir, "metadata.jsonl")
        if not os.path.exists(metadata_path):
            raise ValueError(f"metadata.jsonl not found in {data_dir}")
        
        data_list = []
        with open(metadata_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    # Convert file_name to full path
                    audio_path = os.path.join(data_dir, data['file_name'])
                    data_list.append({
                        self.audio_column: audio_path,
                        self.target_column: data['label']
                    })
        
        return pd.DataFrame(data_list)

    def prepare_columns(self, train_df, valid_df):
        """
        Prepares and standardizes column names for the datasets.

        Args:
            train_df (pd.DataFrame): Training DataFrame.
            valid_df (pd.DataFrame): Validation DataFrame.

        Returns:
            tuple: A tuple containing (train_df, valid_df) with standardized column names.
        """
        # Rename columns to standard format
        column_mapping = {
            self.audio_column: "autotrain_audio",
            self.target_column: "autotrain_label"
        }

        train_df = train_df.rename(columns=column_mapping)
        valid_df = valid_df.rename(columns=column_mapping)

        # Keep only the required columns
        required_columns = ["autotrain_audio", "autotrain_label"]
        train_df = train_df[required_columns]
        valid_df = valid_df[required_columns]

        # Ensure target labels are properly formatted
        # Convert labels to categorical if they are strings
        if train_df["autotrain_label"].dtype == 'object':
            # Get unique labels from both train and validation sets
            train_labels = set(train_df["autotrain_label"].unique())
            valid_labels = set(valid_df["autotrain_label"].unique())
            all_unique_labels = sorted(train_labels.union(valid_labels))
            label_to_id = {label: idx for idx, label in enumerate(all_unique_labels)}
            
            train_df["autotrain_label"] = train_df["autotrain_label"].map(label_to_id)
            valid_df["autotrain_label"] = valid_df["autotrain_label"].map(label_to_id)
            
            logger.info(f"Label mapping: {label_to_id}")
        
        logger.info(f"Final training columns: {list(train_df.columns)}")
        logger.info(f"Final validation columns: {list(valid_df.columns)}")

        return train_df, valid_df

    def prepare(self):
        """
        Main preprocessing method that prepares the complete dataset.

        Returns:
            str: Path to the prepared dataset (local path or Hugging Face Hub dataset ID).
        """
        train_df, valid_df = self.split()
        train_df, valid_df = self.prepare_columns(train_df, valid_df)

        # Convert to Hugging Face Dataset
        train_dataset = Dataset.from_pandas(train_df)
        valid_dataset = Dataset.from_pandas(valid_df)

        # Create DatasetDict
        dataset_dict = DatasetDict({
            "train": train_dataset,
            "validation": valid_dataset
        })

        if self.local:
            # Save dataset locally
            output_path = f"{self.project_name}/autotrain-data"
            os.makedirs(output_path, exist_ok=True)
            dataset_dict.save_to_disk(output_path)
            logger.info(f"Dataset saved locally to: {output_path}")
            return output_path
        else:
            # Upload to Hugging Face Hub
            if not self.username or not self.token:
                raise ValueError("Username and token are required for uploading to Hugging Face Hub")
            
            hub_dataset_id = f"{self.username}/autotrain-data-{self.project_name}"
            
            # Push train split
            train_dataset.push_to_hub(
                hub_dataset_id,
                split="train",
                private=True,
                token=self.token
            )
            
            # Push validation split
            valid_dataset.push_to_hub(
                hub_dataset_id,
                split="validation",
                private=True,
                token=self.token
            )
            
            logger.info(f"Dataset uploaded to Hugging Face Hub: {hub_dataset_id}")
            return hub_dataset_id

@dataclass
class AudioDetectionPreprocessor:
    """
    A preprocessor for audio detection datasets.
    
    Converts audio files with event annotations to temporal event predictions.
    Expected formats:
    - ZIP file with audio files + metadata.jsonl 
    - CSV format with audio_column and events_column
    
    Events format: [{"start": 4.23, "end": 4.27, "label": "car_crash"}, ...]

    Attributes:
        train_data (Union[str, pd.DataFrame]): Path to training data file or DataFrame.
        valid_data (Optional[Union[str, pd.DataFrame]]): Path to validation data file or DataFrame.
        test_data (Optional[Union[str, pd.DataFrame]]): Path to test data file or DataFrame.
        token (Optional[str]): Hugging Face Hub token for uploading datasets.
        project_name (str): Name of the project for output directory.
        username (Optional[str]): Hugging Face username for uploading datasets.
        valid_split (float): Proportion of training data to use for validation if no validation data is provided.
        seed (int): Random seed for reproducible train/validation splits.
        local (bool): Whether to save the dataset locally or upload to Hugging Face Hub.
        audio_column (str): Name of the column containing audio file paths.
        events_column (str): Name of the column containing event annotations.
    """
    
    train_data: Union[str, pd.DataFrame]
    valid_data: Optional[Union[str, pd.DataFrame]] = None
    test_data: Optional[Union[str, pd.DataFrame]] = None
    token: Optional[str] = None
    project_name: str = "project-name"
    username: Optional[str] = None
    valid_split: float = 0.2
    seed: int = 42
    local: bool = True
    audio_column: str = "audio"
    events_column: str = "events"

    def split(self):
        """
        Splits the training data into training and validation sets if no validation data is provided.

        Returns:
            tuple: A tuple containing (train_df, valid_df) DataFrames.
        """
        # Handle ZIP files with metadata.jsonl
        if isinstance(self.train_data, str) and self.train_data.endswith('.zip'):
            # Process ZIP file - extract and read metadata.jsonl
            return self._process_zip_data()
        
        # Handle directory with metadata.jsonl  
        if isinstance(self.train_data, str) and os.path.isdir(self.train_data):
            metadata_path = os.path.join(self.train_data, "metadata.jsonl")
            if os.path.exists(metadata_path):
                return self._process_directory_data()
        
        # Handle regular file-based data
        if isinstance(self.train_data, pd.DataFrame):
            train_df = self.train_data.copy()
        else:
            # Load from file path
            if self.train_data.endswith(('.csv', '.tsv')):
                separator = '\t' if self.train_data.endswith('.tsv') else ','
                train_df = pd.read_csv(self.train_data, sep=separator)
            elif self.train_data.endswith('.jsonl'):
                train_df = pd.read_json(self.train_data, lines=True)
            elif self.train_data.endswith('.json'):
                train_df = pd.read_json(self.train_data)
            else:
                raise ValueError(f"Unsupported file format: {self.train_data}")

        # Process validation data if provided
        if self.valid_data is not None:
            if isinstance(self.valid_data, pd.DataFrame):
                valid_df = self.valid_data.copy()
            else:
                if self.valid_data.endswith(('.csv', '.tsv')):
                    separator = '\t' if self.valid_data.endswith('.tsv') else ','
                    valid_df = pd.read_csv(self.valid_data, sep=separator)
                elif self.valid_data.endswith('.jsonl'):
                    valid_df = pd.read_json(self.valid_data, lines=True)
                elif self.valid_data.endswith('.json'):
                    valid_df = pd.read_json(self.valid_data)
                else:
                    raise ValueError(f"Unsupported file format: {self.valid_data}")
        else:
            # Split training data
            train_df, valid_df = train_test_split(
                train_df, test_size=self.valid_split, random_state=self.seed, stratify=None
            )

        return train_df, valid_df

    def _process_zip_data(self):
        """Process ZIP file containing audio files and metadata.jsonl"""
        import tempfile
        import zipfile
        import json
        import shutil
        
        # Create temporary directory for extraction
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Extract ZIP file
            with zipfile.ZipFile(self.train_data, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # Read metadata.jsonl
            metadata_path = os.path.join(temp_dir, "metadata.jsonl")
            if not os.path.exists(metadata_path):
                raise ValueError("metadata.jsonl not found in ZIP file")
            
            # Load metadata
            data_rows = []
            with open(metadata_path, 'r') as f:
                for line in f:
                    data = json.loads(line.strip())
                    # Convert audio path to full path
                    audio_path = os.path.join(temp_dir, data['file_name'])
                    if os.path.exists(audio_path):
                        data_rows.append({
                            self.audio_column: audio_path,
                            self.events_column: data['events']
                        })
            
            train_df = pd.DataFrame(data_rows)
            
            # Split into train/validation
            train_df, valid_df = train_test_split(
                train_df, test_size=self.valid_split, random_state=self.seed
            )
            
            return train_df, valid_df
            
        except Exception as e:
            raise e
        finally:
            # Always clean up temp directory
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _process_directory_data(self):
        """Process directory containing audio files and metadata.jsonl"""
        import json
        
        metadata_path = os.path.join(self.train_data, "metadata.jsonl")
        
        # Load metadata
        data_rows = []
        with open(metadata_path, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                # Convert audio path to full path
                audio_path = os.path.join(self.train_data, data['file_name'])
                if os.path.exists(audio_path):
                    data_rows.append({
                        self.audio_column: audio_path,
                        self.events_column: data['events']
                    })
        
        train_df = pd.DataFrame(data_rows)
        
        # Handle validation data
        if self.valid_data and os.path.isdir(self.valid_data):
            valid_metadata_path = os.path.join(self.valid_data, "metadata.jsonl")
            if os.path.exists(valid_metadata_path):
                valid_rows = []
                with open(valid_metadata_path, 'r') as f:
                    for line in f:
                        data = json.loads(line.strip())
                        audio_path = os.path.join(self.valid_data, data['file_name'])
                        if os.path.exists(audio_path):
                            valid_rows.append({
                                self.audio_column: audio_path,
                                self.events_column: data['events']
                            })
                valid_df = pd.DataFrame(valid_rows)
            else:
                # Split training data
                train_df, valid_df = train_test_split(
                    train_df, test_size=self.valid_split, random_state=self.seed
                )
        else:
            # Split training data
            train_df, valid_df = train_test_split(
                train_df, test_size=self.valid_split, random_state=self.seed
            )
        
        return train_df, valid_df

    def prepare_columns(self, train_df, valid_df):
        """
        Prepares the column names for the datasets.

        Args:
            train_df (pd.DataFrame): Training DataFrame.
            valid_df (pd.DataFrame): Validation DataFrame.

        Returns:
            tuple: A tuple containing the prepared training and validation DataFrames.
        """
        train_df.loc[:, "autotrain_audio"] = train_df[self.audio_column]
        train_df.loc[:, "autotrain_events"] = train_df[self.events_column]
        
        valid_df.loc[:, "autotrain_audio"] = valid_df[self.audio_column]
        valid_df.loc[:, "autotrain_events"] = valid_df[self.events_column]
        
        # Drop original columns if they have different names
        cols_to_drop = []
        if self.audio_column != "autotrain_audio":
            cols_to_drop.append(self.audio_column)
        if self.events_column != "autotrain_events":
            cols_to_drop.append(self.events_column)
            
        if cols_to_drop:
            train_df = train_df.drop(columns=cols_to_drop)
            valid_df = valid_df.drop(columns=cols_to_drop)
        
        return train_df, valid_df

    def prepare(self):
        """
        Main preprocessing method that prepares the complete audio detection dataset.

        Returns:
            str: Path to the prepared dataset or HuggingFace Hub dataset ID.
        """
        train_df, valid_df = self.split()
        train_df, valid_df = self.prepare_columns(train_df, valid_df)

        if self.local:
            # Save locally
            dataset_dict = DatasetDict({
                "train": Dataset.from_pandas(train_df),
                "validation": Dataset.from_pandas(valid_df)
            })
            
            dataset_path = f"{self.project_name}/autotrain-data"
            dataset_dict.save_to_disk(dataset_path)
            return dataset_path
        else:
            # Upload to Hugging Face Hub
            dataset_dict = DatasetDict({
                "train": Dataset.from_pandas(train_df),
                "validation": Dataset.from_pandas(valid_df)
            })
            
            hub_dataset_id = f"{self.username}/autotrain-data-{self.project_name}"
            dataset_dict.push_to_hub(
                hub_dataset_id,
                token=self.token,
                commit_message="Upload audio detection dataset via AutoTrain"
            )
            return hub_dataset_id


@dataclass
class AudioSegmentationPreprocessor:
    """
    A preprocessor for audio segmentation datasets.
    
    Converts audio files with segment annotations to frame-level predictions.
    Expected CSV format:
    - audio_column: path to audio file
    - segments_column: JSON string with segments like [{"start": 0.0, "end": 1.0, "label": "speech"}, ...]
    OR separate columns for start_time, end_time, label (multiple rows per audio file)

    Attributes:
        train_data (Union[str, pd.DataFrame]): Path to training data file or DataFrame.
        valid_data (Optional[Union[str, pd.DataFrame]]): Path to validation data file or DataFrame.
        test_data (Optional[Union[str, pd.DataFrame]]): Path to test data file or DataFrame.
        token (Optional[str]): Hugging Face Hub token for uploading datasets.
        project_name (str): Name of the project for output directory.
        username (Optional[str]): Hugging Face username for uploading datasets.
        valid_split (float): Proportion of training data to use for validation if no validation data is provided.
        seed (int): Random seed for reproducible train/validation splits.
        local (bool): Whether to save the dataset locally or upload to Hugging Face Hub.
        audio_column (str): Name of the column containing audio file paths.
        segments_column (str): Name of the column containing segment annotations (JSON format).
        frame_rate (int): Frame rate for segmentation (frames per second).
        default_label (str): Default label for unlabeled segments.
    """
    
    train_data: Union[str, pd.DataFrame]
    valid_data: Optional[Union[str, pd.DataFrame]] = None
    test_data: Optional[Union[str, pd.DataFrame]] = None
    token: Optional[str] = None
    project_name: str = "project-name"
    username: Optional[str] = None
    valid_split: float = 0.2
    seed: int = 42
    local: bool = True
    audio_column: str = "audio"
    segments_column: str = "segments"
    frame_rate: int = 100  # 100 fps = 10ms frames
    default_label: str = "silence"
    
    def __post_init__(self):
        """Post-initialization validation and setup."""
        # Validate train_data
        if isinstance(self.train_data, str):
            if not os.path.exists(self.train_data):
                raise ValueError(f"Training data path does not exist: {self.train_data}")
        elif not isinstance(self.train_data, pd.DataFrame):
            raise ValueError(f"Training data must be a string path or pandas DataFrame, got: {type(self.train_data)}")
        
        # Validate valid_data if provided
        if self.valid_data:
            if isinstance(self.valid_data, str):
                if not os.path.exists(self.valid_data):
                    raise ValueError(f"Validation data path does not exist: {self.valid_data}")
            elif not isinstance(self.valid_data, pd.DataFrame):
                raise ValueError(f"Validation data must be a string path or pandas DataFrame, got: {type(self.valid_data)}")

    def _create_frame_labels(self, audio_path, segments, audio_duration=None):
        """
        Create frame-level labels from segment annotations.
        
        Args:
            audio_path (str): Path to audio file
            segments (list): List of segment dictionaries with start, end, label
            audio_duration (float): Duration of audio in seconds (auto-detected if None)
            
        Returns:
            list: Frame-level labels
        """
        import librosa
        import json
        
        # Load audio to get duration if not provided
        if audio_duration is None:
            try:
                y, sr = librosa.load(audio_path, sr=None)
                audio_duration = len(y) / sr
            except Exception as e:
                logger.warning(f"Could not load audio file {audio_path}: {e}")
                audio_duration = 10.0  # Default fallback
        
        # Calculate total frames
        total_frames = int(audio_duration * self.frame_rate)
        
        # Initialize all frames with default label
        frame_labels = [self.default_label] * total_frames
        
        # Parse segments if string
        if isinstance(segments, str):
            try:
                segments = json.loads(segments)
            except:
                logger.warning(f"Could not parse segments JSON: {segments}")
                segments = []
        elif not isinstance(segments, list):
            logger.warning(f"Segments must be a list or JSON string, got: {type(segments)}")
            segments = []
        
        # Fill in segment labels
        for segment in segments:
            start_frame = int(segment['start'] * self.frame_rate)
            end_frame = int(segment['end'] * self.frame_rate)
            label = segment['label']
            
            # Ensure frames are within bounds
            start_frame = max(0, min(start_frame, total_frames - 1))
            end_frame = max(0, min(end_frame, total_frames))
            
            # Fill frames
            for i in range(start_frame, end_frame):
                frame_labels[i] = label
        
        return frame_labels

    def split(self):
        """
        Splits the training data into training and validation sets if no validation data is provided.

        Returns:
            tuple: A tuple containing (train_df, valid_df) DataFrames.
        """
        # Load training data
        if isinstance(self.train_data, pd.DataFrame):
            train_df = self.train_data.copy()
        else:
            # Check if it's a directory with metadata.jsonl (ZIP format)
            if os.path.isdir(self.train_data):
                train_df = self._load_from_metadata_jsonl(self.train_data)
            # Load from file path (CSV/JSONL format)
            elif self.train_data.endswith(('.csv', '.tsv')):
                separator = '\t' if self.train_data.endswith('.tsv') else ','
                train_df = pd.read_csv(self.train_data, sep=separator)
            elif self.train_data.endswith('.jsonl'):
                train_df = pd.read_json(self.train_data, lines=True)
            elif self.train_data.endswith('.json'):
                train_df = pd.read_json(self.train_data)
            else:
                raise ValueError(f"Unsupported file format for training data: {self.train_data}")

        # Load validation data if provided
        if self.valid_data:
            if isinstance(self.valid_data, pd.DataFrame):
                valid_df = self.valid_data.copy()
            else:
                # Check if it's a directory with metadata.jsonl (ZIP format)
                if os.path.isdir(self.valid_data):
                    valid_df = self._load_from_metadata_jsonl(self.valid_data)
                # Load from file path (CSV/JSONL format)
                elif self.valid_data.endswith(('.csv', '.tsv')):
                    separator = '\t' if self.valid_data.endswith('.tsv') else ','
                    valid_df = pd.read_csv(self.valid_data, sep=separator)
                elif self.valid_data.endswith('.jsonl'):
                    valid_df = pd.read_json(self.valid_data, lines=True)
                elif self.valid_data.endswith('.json'):
                    valid_df = pd.read_json(self.valid_data)
                else:
                    raise ValueError(f"Unsupported file format for validation data: {self.valid_data}")
        else:
            # Split training data into train and validation
            train_df, valid_df = train_test_split(
                train_df,
                test_size=self.valid_split,
                random_state=self.seed
            )

        logger.info(f"Training data shape: {train_df.shape}")
        logger.info(f"Validation data shape: {valid_df.shape}")

        return train_df, valid_df

    def _load_from_metadata_jsonl(self, data_dir):
        """
        Load data from a directory containing audio files and metadata.jsonl
        
        Args:
            data_dir (str): Directory containing audio files and metadata.jsonl
            
        Returns:
            pd.DataFrame: DataFrame with file_name and segments columns
        """
        import json
        
        metadata_path = os.path.join(data_dir, "metadata.jsonl")
        if not os.path.exists(metadata_path):
            raise ValueError(f"metadata.jsonl not found in {data_dir}")
        
        data_list = []
        with open(metadata_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    # Convert file_name to full path
                    audio_path = os.path.join(data_dir, data['file_name'])
                    data_list.append({
                        self.audio_column: audio_path,
                        self.segments_column: data['segments']
                    })
        
        return pd.DataFrame(data_list)

    def prepare_columns(self, train_df, valid_df):
        """
        Prepares and standardizes column names for the datasets.
        Converts segment annotations to frame-level labels.

        Args:
            train_df (pd.DataFrame): Training DataFrame.
            valid_df (pd.DataFrame): Validation DataFrame.

        Returns:
            tuple: A tuple containing (train_df, valid_df) with frame-level labels.
        """
        processed_train = []
        processed_valid = []
        
        # Get unique labels for mapping
        all_labels = set([self.default_label])
        
        # Collect all labels first
        for df in [train_df, valid_df]:
            for _, row in df.iterrows():
                segments = row[self.segments_column]
                if isinstance(segments, str):
                    try:
                        import json
                        segments = json.loads(segments)
                        for segment in segments:
                            all_labels.add(segment['label'])
                    except:
                        pass
                elif isinstance(segments, list):
                    # Handle list of dictionaries directly
                    for segment in segments:
                        if isinstance(segment, dict) and 'label' in segment:
                            all_labels.add(segment['label'])
        
        # Create label to ID mapping
        label_list = sorted(list(all_labels))
        label_to_id = {label: idx for idx, label in enumerate(label_list)}
        logger.info(f"Label mapping: {label_to_id}")
        
        # Process training data
        for _, row in train_df.iterrows():
            audio_path = row[self.audio_column]
            segments = row[self.segments_column]
            
            frame_labels = self._create_frame_labels(audio_path, segments)
            frame_ids = [label_to_id[label] for label in frame_labels]
            
            processed_train.append({
                "autotrain_audio": audio_path,
                "autotrain_label": frame_ids
            })
        
        # Process validation data
        for _, row in valid_df.iterrows():
            audio_path = row[self.audio_column]
            segments = row[self.segments_column]
            
            frame_labels = self._create_frame_labels(audio_path, segments)
            frame_ids = [label_to_id[label] for label in frame_labels]
            
            processed_valid.append({
                "autotrain_audio": audio_path,
                "autotrain_label": frame_ids
            })
        
        train_df_processed = pd.DataFrame(processed_train)
        valid_df_processed = pd.DataFrame(processed_valid)
        
        logger.info(f"Processed training data shape: {train_df_processed.shape}")
        logger.info(f"Processed validation data shape: {valid_df_processed.shape}")
        logger.info(f"Frame labels example length: {len(processed_train[0]['autotrain_label']) if processed_train else 0}")

        return train_df_processed, valid_df_processed

    def prepare(self):
        """
        Main preprocessing method that prepares the complete dataset.

        Returns:
            str: Path to the prepared dataset (local path or Hugging Face Hub dataset ID).
        """
        train_df, valid_df = self.split()
        train_df, valid_df = self.prepare_columns(train_df, valid_df)

        # Convert to Hugging Face Dataset
        train_dataset = Dataset.from_pandas(train_df)
        valid_dataset = Dataset.from_pandas(valid_df)

        # Create DatasetDict
        dataset_dict = DatasetDict({
            "train": train_dataset,
            "validation": valid_dataset
        })

        if self.local:
            # Save dataset locally
            output_path = f"{self.project_name}/autotrain-data"
            os.makedirs(output_path, exist_ok=True)
            dataset_dict.save_to_disk(output_path)
            logger.info(f"Audio segmentation dataset saved locally to: {output_path}")
            return output_path
        else:
            # Upload to Hugging Face Hub
            if not self.username or not self.token:
                raise ValueError("Username and token are required for uploading to Hugging Face Hub")
            
            hub_dataset_id = f"{self.username}/autotrain-data-{self.project_name}"
            
            # Push train split
            train_dataset.push_to_hub(
                hub_dataset_id,
                split="train",
                private=True,
                token=self.token
            )
            
            # Push validation split
            valid_dataset.push_to_hub(
                hub_dataset_id,
                split="validation",
                private=True,
                token=self.token
            )
            
            logger.info(f"Audio segmentation dataset uploaded to Hugging Face Hub: {hub_dataset_id}")
            return hub_dataset_id 