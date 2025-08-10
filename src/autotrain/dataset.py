import io
import os
import uuid
import zipfile
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

from autotrain.preprocessor.audio import (
    AudioClassificationPreprocessor, 
    AudioDetectionPreprocessor, 
    AudioSegmentationPreprocessor
)

from autotrain.preprocessor.tabular import (
    TabularBinaryClassificationPreprocessor,
    TabularMultiClassClassificationPreprocessor,
    TabularMultiColumnRegressionPreprocessor,
    TabularMultiLabelClassificationPreprocessor,
    TabularSingleColumnRegressionPreprocessor,
)
from autotrain.preprocessor.text import (
    LLMPreprocessor,
    SentenceTransformersPreprocessor,
    Seq2SeqPreprocessor,
    TextBinaryClassificationPreprocessor,
    TextExtractiveQuestionAnsweringPreprocessor,
    TextMultiClassClassificationPreprocessor,
    TextSingleColumnRegressionPreprocessor,
    TextTokenClassificationPreprocessor,
)
from autotrain.preprocessor.vision import (
    ImageClassificationPreprocessor,
    ImageInstanceSegmentationPreprocessor,
    ImageRegressionPreprocessor,
    ImageSemanticSegmentationPreprocessor,
    ObjectDetectionPreprocessor,
)
from autotrain.preprocessor.vlm import VLMPreprocessor


def remove_non_image_files(folder):
    """
    Remove non-image files from a specified folder and its subfolders.

    This function iterates through all files in the given folder and its subfolders,
    and removes any file that does not have an allowed image file extension. The allowed
    extensions are: .jpg, .jpeg, .png, .JPG, .JPEG, .PNG, and .jsonl.

    Args:
        folder (str): The path to the folder from which non-image files should be removed.

    Returns:
        None
    """
    allowed_extensions = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG", ".jsonl", ".txt"}

    for root, dirs, files in os.walk(folder):
        for file in files:
            file_extension = os.path.splitext(file)[1]

            if file_extension.lower() not in allowed_extensions:
                file_path = os.path.join(root, file)
                os.remove(file_path)
                print(f"Removed file: {file_path}")

        for subfolder in dirs:
            remove_non_image_files(os.path.join(root, subfolder))


def remove_non_audio_files(folder):
    """
    Remove non-audio files from a specified folder and its subfolders.

    This function iterates through all files in the given folder and its subfolders,
    and removes any file that does not have an allowed audio file extension. The allowed
    extensions are: .wav, .mp3, .flac, .m4a, .ogg, .aac, .jsonl.

    Args:
        folder (str): The path to the folder from which non-audio files should be removed.

    Returns:
        None
    """
    allowed_extensions = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac", ".WAV", ".MP3", ".FLAC", ".M4A", ".OGG", ".AAC", ".jsonl", ".txt"}

    for root, dirs, files in os.walk(folder):
        for file in files:
            file_extension = os.path.splitext(file)[1]

            if file_extension.lower() not in allowed_extensions:
                file_path = os.path.join(root, file)
                os.remove(file_path)
                print(f"Removed file: {file_path}")

        for subfolder in dirs:
            remove_non_audio_files(os.path.join(root, subfolder))


def extract_zip_file(zip_path, extract_dir, task_type="general"):
    """
    Extract ZIP file and handle nested directory structures.
    
    Args:
        zip_path (str): Path to the ZIP file
        extract_dir (str): Directory to extract to
        task_type (str): Type of task to detect appropriate data patterns
                        ("classification", "segmentation", "detection", "regression", "vlm", "audio", "general")
        
    Returns:
        str: Path to the actual data directory (handles nested structures)
    """
    import zipfile
    
    os.makedirs(extract_dir, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    
    macosx_dir = os.path.join(extract_dir, "__MACOSX")
    if os.path.exists(macosx_dir):
        os.system(f"rm -rf {macosx_dir}")
    
    # Clean up non-relevant files based on task type
    if task_type == "audio":
        remove_non_audio_files(extract_dir)
    else:
        # Default to image cleanup for vision tasks
        remove_non_image_files(extract_dir)
    
    subdirs = [d for d in os.listdir(extract_dir) if os.path.isdir(os.path.join(extract_dir, d))]
    if len(subdirs) == 1:
        potential_data_dir = os.path.join(extract_dir, subdirs[0])
        subdir_contents = os.listdir(potential_data_dir)
        
        if task_type == "classification":
            class_dirs = [d for d in subdir_contents if os.path.isdir(os.path.join(potential_data_dir, d))]
            if len(class_dirs) >= 2:
                return potential_data_dir
        elif task_type == "segmentation":
            data_indicators = ['images', 'img', 'masks', 'mask', 'annotations', 'classes.txt']
            if any(item in subdir_contents for item in data_indicators):
                return potential_data_dir
        elif task_type in ["detection", "regression", "vlm"]:
            has_metadata = 'metadata.jsonl' in subdir_contents
            has_images = any(f.lower().endswith(('.jpg', '.jpeg', '.png')) for f in subdir_contents)
            if has_metadata and has_images:
                return potential_data_dir
        else:
            data_indicators = ['images', 'img', 'masks', 'mask', 'annotations', 'classes.txt', 'metadata.jsonl']
            has_data_structure = any(item in subdir_contents for item in data_indicators)
            class_dirs = [d for d in subdir_contents if os.path.isdir(os.path.join(potential_data_dir, d))]
            has_class_structure = len(class_dirs) >= 2
            has_metadata = 'metadata.jsonl' in subdir_contents
            has_images = any(f.lower().endswith(('.jpg', '.jpeg', '.png')) for f in subdir_contents)
            has_detection_structure = has_metadata and has_images
            
            if has_data_structure or has_class_structure or has_detection_structure:
                return potential_data_dir
    
    return extract_dir


@dataclass
class AutoTrainImageClassificationDataset:
    """
    A class to handle image classification datasets for AutoTrain.

    Attributes:
        train_data (str): Path to the training data.
        token (str): Authentication token.
        project_name (str): Name of the project.
        username (str): Username of the project owner.
        valid_data (Optional[str]): Path to the validation data. Default is None.
        percent_valid (Optional[float]): Percentage of training data to use for validation. Default is None.
        local (bool): Flag to indicate if the data is local. Default is False.

    Methods:
        __str__() -> str:
            Returns a string representation of the dataset.

        __post_init__():
            Initializes the dataset and sets default values for validation data.

        prepare():
            Prepares the dataset for training by extracting and preprocessing the data.
    """

    train_data: str
    token: str
    project_name: str
    username: str
    valid_data: Optional[str] = None
    percent_valid: Optional[float] = None
    local: bool = False

    def __str__(self) -> str:
        info = f"Dataset: {self.project_name} ({self.task})\n"
        info += f"Train data: {self.train_data}\n"
        info += f"Valid data: {self.valid_data}\n"
        return info

    def __post_init__(self):
        self.task = "image_multi_class_classification"
        if not self.valid_data and self.percent_valid is None:
            self.percent_valid = 0.2
        elif self.valid_data and self.percent_valid is not None:
            raise ValueError("You can only specify one of valid_data or percent_valid")
        elif self.valid_data:
            self.percent_valid = 0.0

    def prepare(self):
        valid_dir = None
        
        if not isinstance(self.train_data, str):
            cache_dir = os.environ.get("HF_HOME")
            if not cache_dir:
                cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")

            random_uuid = uuid.uuid4()
            train_dir = os.path.join(cache_dir, "autotrain", str(random_uuid))
            os.makedirs(train_dir, exist_ok=True)
            self.train_data.seek(0)
            content = self.train_data.read()
            bytes_io = io.BytesIO(content)

            zip_ref = zipfile.ZipFile(bytes_io, "r")
            zip_ref.extractall(train_dir)
            macosx_dir = os.path.join(train_dir, "__MACOSX")
            if os.path.exists(macosx_dir):
                os.system(f"rm -rf {macosx_dir}")
            remove_non_image_files(train_dir)
            
            if self.valid_data:
                random_uuid = uuid.uuid4()
                valid_dir = os.path.join(cache_dir, "autotrain", str(random_uuid))
                os.makedirs(valid_dir, exist_ok=True)
                self.valid_data.seek(0)
                content = self.valid_data.read()
                bytes_io = io.BytesIO(content)
                zip_ref = zipfile.ZipFile(bytes_io, "r")
                zip_ref.extractall(valid_dir)
                macosx_dir = os.path.join(valid_dir, "__MACOSX")
                if os.path.exists(macosx_dir):
                    os.system(f"rm -rf {macosx_dir}")
                remove_non_image_files(valid_dir)
                
        elif isinstance(self.train_data, str) and self.train_data.endswith('.zip') and os.path.isfile(self.train_data):
            cache_dir = os.environ.get("HF_HOME")
            if not cache_dir:
                cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")

            random_uuid = uuid.uuid4()
            extract_dir = os.path.join(cache_dir, "autotrain", str(random_uuid))
            train_dir = extract_zip_file(self.train_data, extract_dir, task_type="classification")
            
            if self.valid_data and self.valid_data.endswith('.zip') and os.path.isfile(self.valid_data):
                random_uuid = uuid.uuid4()
                valid_extract_dir = os.path.join(cache_dir, "autotrain", str(random_uuid))
                valid_dir = extract_zip_file(self.valid_data, valid_extract_dir, task_type="classification")
            elif self.valid_data:
                valid_dir = self.valid_data
                
        else:
            train_dir = self.train_data
            if self.valid_data:
                valid_dir = self.valid_data

        preprocessor = ImageClassificationPreprocessor(
            train_data=train_dir,
            valid_data=valid_dir,
            token=self.token,
            project_name=self.project_name,
            username=self.username,
            local=self.local,
        )
        return preprocessor.prepare()


@dataclass  
class AutoTrainImageSemanticSegmentationDataset:
    """
    A class to handle image semantic segmentation datasets for AutoTrain.

    Attributes:
        train_data (str): Path to the training data.
        token (str): Authentication token.
        project_name (str): Name of the project.
        username (str): Username of the project owner.
        valid_data (Optional[str]): Path to the validation data. Default is None.
        percent_valid (Optional[float]): Percentage of training data to use for validation. Default is None.
        local (bool): Flag to indicate if the data is local. Default is False.

    Methods:
        __str__() -> str:
            Returns a string representation of the dataset.

        __post_init__():
            Initializes the dataset and sets default values for validation data.

        prepare():
            Prepares the dataset for training by extracting and preprocessing the data.
    """

    train_data: str
    token: str
    project_name: str
    username: str
    valid_data: Optional[str] = None
    percent_valid: Optional[float] = None
    local: bool = False

    def __str__(self) -> str:
        info = f"Dataset: {self.project_name} ({self.task})\n"
        info += f"Train data: {self.train_data}\n"
        info += f"Valid data: {self.valid_data}\n"
        return info

    def __post_init__(self):
        self.task = "image_semantic_segmentation"
        if not self.valid_data and self.percent_valid is None:
            self.percent_valid = 0.2
        elif self.valid_data and self.percent_valid is not None:
            raise ValueError("You can only specify one of valid_data or percent_valid")
        elif self.valid_data:
            self.percent_valid = 0.0

    def prepare(self):
        valid_dir = None
        
        if not isinstance(self.train_data, str):
            cache_dir = os.environ.get("HF_HOME")
            if not cache_dir:
                cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")

            random_uuid = uuid.uuid4()
            train_dir = os.path.join(cache_dir, "autotrain", str(random_uuid))
            os.makedirs(train_dir, exist_ok=True)
            self.train_data.seek(0)
            content = self.train_data.read()
            bytes_io = io.BytesIO(content)

            zip_ref = zipfile.ZipFile(bytes_io, "r")
            zip_ref.extractall(train_dir)
            macosx_dir = os.path.join(train_dir, "__MACOSX")
            if os.path.exists(macosx_dir):
                os.system(f"rm -rf {macosx_dir}")
            remove_non_image_files(train_dir)
            
            if self.valid_data:
                random_uuid = uuid.uuid4()
                valid_dir = os.path.join(cache_dir, "autotrain", str(random_uuid))
                os.makedirs(valid_dir, exist_ok=True)
                self.valid_data.seek(0)
                content = self.valid_data.read()
                bytes_io = io.BytesIO(content)
                zip_ref = zipfile.ZipFile(bytes_io, "r")
                zip_ref.extractall(valid_dir)
                macosx_dir = os.path.join(valid_dir, "__MACOSX")
                if os.path.exists(macosx_dir):
                    os.system(f"rm -rf {macosx_dir}")
                remove_non_image_files(valid_dir)
                
        elif isinstance(self.train_data, str) and self.train_data.endswith('.zip') and os.path.isfile(self.train_data):
            cache_dir = os.environ.get("HF_HOME")
            if not cache_dir:
                cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")

            random_uuid = uuid.uuid4()
            extract_dir = os.path.join(cache_dir, "autotrain", str(random_uuid))
            train_dir = extract_zip_file(self.train_data, extract_dir, task_type="segmentation")
            
            if self.valid_data and self.valid_data.endswith('.zip') and os.path.isfile(self.valid_data):
                random_uuid = uuid.uuid4()
                valid_extract_dir = os.path.join(cache_dir, "autotrain", str(random_uuid))
                valid_dir = extract_zip_file(self.valid_data, valid_extract_dir, task_type="segmentation")
            elif self.valid_data:
                valid_dir = self.valid_data
                
        else:
            train_dir = self.train_data
            if self.valid_data:
                valid_dir = self.valid_data

        preprocessor = ImageSemanticSegmentationPreprocessor(
            train_data=train_dir,
            valid_data=valid_dir,
            token=self.token,
            project_name=self.project_name,
            username=self.username,
            local=self.local,
        )
        return preprocessor.prepare()


@dataclass  
class AutoTrainImageInstanceSegmentationDataset:
    """
    A class to handle image instance segmentation datasets for AutoTrain.

    Attributes:
        train_data (str): Path to the training data.
        token (str): Authentication token.
        project_name (str): Name of the project.
        username (str): Username of the project owner.
        valid_data (Optional[str]): Path to the validation data. Default is None.
        percent_valid (Optional[float]): Percentage of training data to use for validation. Default is None.
        local (bool): Flag to indicate if the data is local. Default is False.

    Methods:
        __str__() -> str:
            Returns a string representation of the dataset.

        __post_init__():
            Initializes the dataset and sets default values for validation data.

        prepare():
            Prepares the dataset for training by extracting and preprocessing the data.
    """

    train_data: str
    token: str
    project_name: str
    username: str
    valid_data: Optional[str] = None
    percent_valid: Optional[float] = None
    local: bool = False

    def __str__(self) -> str:
        info = f"Dataset: {self.project_name} ({self.task})\n"
        info += f"Train data: {self.train_data}\n"
        info += f"Valid data: {self.valid_data}\n"
        return info

    def __post_init__(self):
        self.task = "image_instance_segmentation"
        if not self.valid_data and self.percent_valid is None:
            self.percent_valid = 0.2
        elif self.valid_data and self.percent_valid is not None:
            raise ValueError("You can only specify one of valid_data or percent_valid")
        elif self.valid_data:
            self.percent_valid = 0.0

    def prepare(self):
        valid_dir = None
        
        if not isinstance(self.train_data, str):
            cache_dir = os.environ.get("HF_HOME")
            if not cache_dir:
                cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")

            random_uuid = uuid.uuid4()
            train_dir = os.path.join(cache_dir, "autotrain", str(random_uuid))
            os.makedirs(train_dir, exist_ok=True)
            self.train_data.seek(0)
            content = self.train_data.read()
            bytes_io = io.BytesIO(content)

            zip_ref = zipfile.ZipFile(bytes_io, "r")
            zip_ref.extractall(train_dir)
            macosx_dir = os.path.join(train_dir, "__MACOSX")
            if os.path.exists(macosx_dir):
                os.system(f"rm -rf {macosx_dir}")
            remove_non_image_files(train_dir)
            
            if self.valid_data:
                random_uuid = uuid.uuid4()
                valid_dir = os.path.join(cache_dir, "autotrain", str(random_uuid))
                os.makedirs(valid_dir, exist_ok=True)
                self.valid_data.seek(0)
                content = self.valid_data.read()
                bytes_io = io.BytesIO(content)
                zip_ref = zipfile.ZipFile(bytes_io, "r")
                zip_ref.extractall(valid_dir)
                # remove the __MACOSX directory
                macosx_dir = os.path.join(valid_dir, "__MACOSX")
                if os.path.exists(macosx_dir):
                    os.system(f"rm -rf {macosx_dir}")
                remove_non_image_files(valid_dir)
                
        elif isinstance(self.train_data, str) and self.train_data.endswith('.zip') and os.path.isfile(self.train_data):
            # CLI: ZIP file path -> extract ZIP
            cache_dir = os.environ.get("HF_HOME")
            if not cache_dir:
                cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")

            random_uuid = uuid.uuid4()
            extract_dir = os.path.join(cache_dir, "autotrain", str(random_uuid))
            train_dir = extract_zip_file(self.train_data, extract_dir, task_type="segmentation")
            
            if self.valid_data and self.valid_data.endswith('.zip') and os.path.isfile(self.valid_data):
                random_uuid = uuid.uuid4()
                valid_extract_dir = os.path.join(cache_dir, "autotrain", str(random_uuid))
                valid_dir = extract_zip_file(self.valid_data, valid_extract_dir, task_type="segmentation")
            elif self.valid_data:
                valid_dir = self.valid_data
                
        else:
            # CLI: Directory path -> use directly
            train_dir = self.train_data
            if self.valid_data:
                valid_dir = self.valid_data

        preprocessor = ImageInstanceSegmentationPreprocessor(
            train_data=train_dir,
            valid_data=valid_dir,
            token=self.token,
            project_name=self.project_name,
            username=self.username,
            local=self.local,
        )
        return preprocessor.prepare()


@dataclass
class AutoTrainObjectDetectionDataset:
    """
    A class to handle object detection datasets for AutoTrain.

    Attributes:
        train_data (str): Path to the training data.
        token (str): Authentication token.
        project_name (str): Name of the project.
        username (str): Username of the project owner.
        valid_data (Optional[str]): Path to the validation data. Default is None.
        percent_valid (Optional[float]): Percentage of training data to use for validation. Default is None.
        local (bool): Flag to indicate if the data is local. Default is False.

    Methods:
        __str__() -> str:
            Returns a string representation of the dataset.

        __post_init__():
            Initializes the dataset and sets default values for validation data.

        prepare():
            Prepares the dataset for training by extracting and preprocessing the data.
    """

    train_data: str
    token: str
    project_name: str
    username: str
    valid_data: Optional[str] = None
    percent_valid: Optional[float] = None
    local: bool = False

    def __str__(self) -> str:
        info = f"Dataset: {self.project_name} ({self.task})\n"
        info += f"Train data: {self.train_data}\n"
        info += f"Valid data: {self.valid_data}\n"
        return info

    def __post_init__(self):
        self.task = "object_detection"
        if not self.valid_data and self.percent_valid is None:
            self.percent_valid = 0.2
        elif self.valid_data and self.percent_valid is not None:
            raise ValueError("You can only specify one of valid_data or percent_valid")
        elif self.valid_data:
            self.percent_valid = 0.0

    def prepare(self):
        valid_dir = None
        
        # Handle different input types
        if not isinstance(self.train_data, str):
            # Web upload: file-like object -> extract ZIP
            cache_dir = os.environ.get("HF_HOME")
            if not cache_dir:
                cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")

            random_uuid = uuid.uuid4()
            train_dir = os.path.join(cache_dir, "autotrain", str(random_uuid))
            os.makedirs(train_dir, exist_ok=True)
            self.train_data.seek(0)
            content = self.train_data.read()
            bytes_io = io.BytesIO(content)

            zip_ref = zipfile.ZipFile(bytes_io, "r")
            zip_ref.extractall(train_dir)
            # remove the __MACOSX directory
            macosx_dir = os.path.join(train_dir, "__MACOSX")
            if os.path.exists(macosx_dir):
                os.system(f"rm -rf {macosx_dir}")
            remove_non_image_files(train_dir)
            
            if self.valid_data:
                random_uuid = uuid.uuid4()
                valid_dir = os.path.join(cache_dir, "autotrain", str(random_uuid))
                os.makedirs(valid_dir, exist_ok=True)
                self.valid_data.seek(0)
                content = self.valid_data.read()
                bytes_io = io.BytesIO(content)
                zip_ref = zipfile.ZipFile(bytes_io, "r")
                zip_ref.extractall(valid_dir)
                # remove the __MACOSX directory
                macosx_dir = os.path.join(valid_dir, "__MACOSX")
                if os.path.exists(macosx_dir):
                    os.system(f"rm -rf {macosx_dir}")
                remove_non_image_files(valid_dir)
                
        elif isinstance(self.train_data, str) and self.train_data.endswith('.zip') and os.path.isfile(self.train_data):
            # CLI: ZIP file path -> extract ZIP
            cache_dir = os.environ.get("HF_HOME")
            if not cache_dir:
                cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")

            random_uuid = uuid.uuid4()
            extract_dir = os.path.join(cache_dir, "autotrain", str(random_uuid))
            train_dir = extract_zip_file(self.train_data, extract_dir, task_type="detection")
            
            if self.valid_data and self.valid_data.endswith('.zip') and os.path.isfile(self.valid_data):
                random_uuid = uuid.uuid4()
                valid_extract_dir = os.path.join(cache_dir, "autotrain", str(random_uuid))
                valid_dir = extract_zip_file(self.valid_data, valid_extract_dir, task_type="detection")
            elif self.valid_data:
                valid_dir = self.valid_data
                
        else:
            # CLI: Directory path -> use directly
            train_dir = self.train_data
            if self.valid_data:
                valid_dir = self.valid_data

        preprocessor = ObjectDetectionPreprocessor(
            train_data=train_dir,
            valid_data=valid_dir,
            token=self.token,
            project_name=self.project_name,
            username=self.username,
            local=self.local,
        )
        return preprocessor.prepare()


@dataclass
class AutoTrainVLMDataset:
    """
    A class to handle VLM (Vision Language Model) datasets for AutoTrain.

    Attributes:
        train_data (str): Path to the training data.
        token (str): Authentication token.
        project_name (str): Name of the project.
        username (str): Username of the project owner.
        column_mapping (dict): Mapping of column names for the dataset.
        valid_data (Optional[str]): Path to the validation data. Default is None.
        percent_valid (Optional[float]): Percentage of training data to use for validation. Default is None.
        local (bool): Flag to indicate if the data is local. Default is False.

    Methods:
        __str__() -> str:
            Returns a string representation of the dataset.

        __post_init__():
            Initializes the dataset and sets default values for validation data.

        prepare():
            Prepares the dataset for training by extracting and preprocessing the data.
    """

    train_data: str
    token: str
    project_name: str
    username: str
    column_mapping: dict
    valid_data: Optional[str] = None
    percent_valid: Optional[float] = None
    local: bool = False

    def __str__(self) -> str:
        info = f"Dataset: {self.project_name} ({self.task})\n"
        info += f"Train data: {self.train_data}\n"
        info += f"Valid data: {self.valid_data}\n"
        return info

    def __post_init__(self):
        self.task = "vlm"
        if not self.valid_data and self.percent_valid is None:
            self.percent_valid = 0.2
        elif self.valid_data and self.percent_valid is not None:
            raise ValueError("You can only specify one of valid_data or percent_valid")
        elif self.valid_data:
            self.percent_valid = 0.0

    def prepare(self):
        valid_dir = None
        
        if not isinstance(self.train_data, str):
            cache_dir = os.environ.get("HF_HOME")
            if not cache_dir:
                cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")

            random_uuid = uuid.uuid4()
            train_dir = os.path.join(cache_dir, "autotrain", str(random_uuid))
            os.makedirs(train_dir, exist_ok=True)
            self.train_data.seek(0)
            content = self.train_data.read()
            bytes_io = io.BytesIO(content)

            zip_ref = zipfile.ZipFile(bytes_io, "r")
            zip_ref.extractall(train_dir)
            macosx_dir = os.path.join(train_dir, "__MACOSX")
            if os.path.exists(macosx_dir):
                os.system(f"rm -rf {macosx_dir}")
            remove_non_image_files(train_dir)
            
            if self.valid_data:
                random_uuid = uuid.uuid4()
                valid_dir = os.path.join(cache_dir, "autotrain", str(random_uuid))
                os.makedirs(valid_dir, exist_ok=True)
                self.valid_data.seek(0)
                content = self.valid_data.read()
                bytes_io = io.BytesIO(content)
                zip_ref = zipfile.ZipFile(bytes_io, "r")
                zip_ref.extractall(valid_dir)
                macosx_dir = os.path.join(valid_dir, "__MACOSX")
                if os.path.exists(macosx_dir):
                    os.system(f"rm -rf {macosx_dir}")
                remove_non_image_files(valid_dir)
                
        elif isinstance(self.train_data, str) and self.train_data.endswith('.zip') and os.path.isfile(self.train_data):
            cache_dir = os.environ.get("HF_HOME")
            if not cache_dir:
                cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")

            random_uuid = uuid.uuid4()
            extract_dir = os.path.join(cache_dir, "autotrain", str(random_uuid))
            train_dir = extract_zip_file(self.train_data, extract_dir, task_type="vlm")
            
            if self.valid_data and self.valid_data.endswith('.zip') and os.path.isfile(self.valid_data):
                random_uuid = uuid.uuid4()
                valid_extract_dir = os.path.join(cache_dir, "autotrain", str(random_uuid))
                valid_dir = extract_zip_file(self.valid_data, valid_extract_dir, task_type="vlm")
            elif self.valid_data:
                valid_dir = self.valid_data
                
        else:
            train_dir = self.train_data
            if self.valid_data:
                valid_dir = self.valid_data

        preprocessor = VLMPreprocessor(
            train_data=train_dir,
            valid_data=valid_dir,
            token=self.token,
            project_name=self.project_name,
            username=self.username,
            column_mapping=self.column_mapping,
            local=self.local,
        )
        return preprocessor.prepare()


@dataclass
class AutoTrainImageRegressionDataset:
    """
    A class to handle image regression datasets for AutoTrain.

    Attributes:
        train_data (str): Path to the training data.
        token (str): Authentication token.
        project_name (str): Name of the project.
        username (str): Username of the project owner.
        valid_data (Optional[str]): Path to the validation data. Default is None.
        percent_valid (Optional[float]): Percentage of training data to use for validation. Default is None.
        local (bool): Flag to indicate if the data is local. Default is False.

    Methods:
        __str__() -> str:
            Returns a string representation of the dataset.

        __post_init__():
            Initializes the dataset and sets default values for validation data.

        prepare():
            Prepares the dataset for training by extracting and preprocessing the data.
    """

    train_data: str
    token: str
    project_name: str
    username: str
    valid_data: Optional[str] = None
    percent_valid: Optional[float] = None
    local: bool = False

    def __str__(self) -> str:
        info = f"Dataset: {self.project_name} ({self.task})\n"
        info += f"Train data: {self.train_data}\n"
        info += f"Valid data: {self.valid_data}\n"
        return info

    def __post_init__(self):
        self.task = "image_regression"
        if not self.valid_data and self.percent_valid is None:
            self.percent_valid = 0.2
        elif self.valid_data and self.percent_valid is not None:
            raise ValueError("You can only specify one of valid_data or percent_valid")
        elif self.valid_data:
            self.percent_valid = 0.0

    def prepare(self):
        valid_dir = None
        
        if not isinstance(self.train_data, str):
            cache_dir = os.environ.get("HF_HOME")
            if not cache_dir:
                cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")

            random_uuid = uuid.uuid4()
            train_dir = os.path.join(cache_dir, "autotrain", str(random_uuid))
            os.makedirs(train_dir, exist_ok=True)
            self.train_data.seek(0)
            content = self.train_data.read()
            bytes_io = io.BytesIO(content)

            zip_ref = zipfile.ZipFile(bytes_io, "r")
            zip_ref.extractall(train_dir)
            macosx_dir = os.path.join(train_dir, "__MACOSX")
            if os.path.exists(macosx_dir):
                os.system(f"rm -rf {macosx_dir}")
            remove_non_image_files(train_dir)
            
            if self.valid_data:
                random_uuid = uuid.uuid4()
                valid_dir = os.path.join(cache_dir, "autotrain", str(random_uuid))
                os.makedirs(valid_dir, exist_ok=True)
                self.valid_data.seek(0)
                content = self.valid_data.read()
                bytes_io = io.BytesIO(content)
                zip_ref = zipfile.ZipFile(bytes_io, "r")
                zip_ref.extractall(valid_dir)
                macosx_dir = os.path.join(valid_dir, "__MACOSX")
                if os.path.exists(macosx_dir):
                    os.system(f"rm -rf {macosx_dir}")
                remove_non_image_files(valid_dir)
                
        elif isinstance(self.train_data, str) and self.train_data.endswith('.zip') and os.path.isfile(self.train_data):
            cache_dir = os.environ.get("HF_HOME")
            if not cache_dir:
                cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")

            random_uuid = uuid.uuid4()
            extract_dir = os.path.join(cache_dir, "autotrain", str(random_uuid))
            train_dir = extract_zip_file(self.train_data, extract_dir, task_type="regression")
            
            if self.valid_data and self.valid_data.endswith('.zip') and os.path.isfile(self.valid_data):
                random_uuid = uuid.uuid4()
                valid_extract_dir = os.path.join(cache_dir, "autotrain", str(random_uuid))
                valid_dir = extract_zip_file(self.valid_data, valid_extract_dir, task_type="regression")
            elif self.valid_data:
                valid_dir = self.valid_data
                
        else:
            train_dir = self.train_data
            if self.valid_data:
                valid_dir = self.valid_data

        preprocessor = ImageRegressionPreprocessor(
            train_data=train_dir,
            valid_data=valid_dir,
            token=self.token,
            project_name=self.project_name,
            username=self.username,
            local=self.local,
        )
        return preprocessor.prepare()


@dataclass
class AutoTrainAudioClassificationDataset:
    """
    A dataset class for AutoTrain audio classification tasks (ZIP format).

    Attributes:
        train_data (str): Path to the training data ZIP file.
        token (str): Authentication token.
        project_name (str): Name of the project.
        username (str): Username of the project owner.
        valid_data (Optional[str]): Path to the validation data ZIP file. Default is None.
        percent_valid (Optional[float]): Percentage of training data to be used for validation. Default is None.
        local (bool): Flag indicating if the data is local. Default is False.

    Methods:
        __str__() -> str:
            Returns a string representation of the dataset.

        __post_init__():
            Initializes the dataset and sets default values for validation data.

        prepare():
            Prepares the dataset for training by extracting and preprocessing the audio data.
    """

    train_data: str
    token: str
    project_name: str
    username: str
    valid_data: Optional[str] = None
    percent_valid: Optional[float] = None
    local: bool = False

    def __str__(self) -> str:
        info = f"Dataset: {self.project_name} ({self.task})\n"
        info += f"Train data: {self.train_data}\n"
        info += f"Valid data: {self.valid_data}\n"
        return info

    def __post_init__(self):
        self.task = "audio_multi_class_classification"
        if not self.valid_data and self.percent_valid is None:
            self.percent_valid = 0.2
        elif self.valid_data and self.percent_valid is not None:
            raise ValueError("You can only specify one of valid_data or percent_valid")
        elif self.valid_data:
            self.percent_valid = 0.0

    def prepare(self):
        valid_dir = None
        if not isinstance(self.train_data, str):
            cache_dir = os.environ.get("HF_HOME")
            if not cache_dir:
                cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")

            random_uuid = uuid.uuid4()
            train_dir = os.path.join(cache_dir, "autotrain", str(random_uuid))
            os.makedirs(train_dir, exist_ok=True)
            self.train_data.seek(0)
            content = self.train_data.read()
            bytes_io = io.BytesIO(content)

            zip_ref = zipfile.ZipFile(bytes_io, "r")
            zip_ref.extractall(train_dir)
            macosx_dir = os.path.join(train_dir, "__MACOSX")
            if os.path.exists(macosx_dir):
                os.system(f"rm -rf {macosx_dir}")
            
            self._validate_audio_data(train_dir)
            
            if self.valid_data:
                random_uuid = uuid.uuid4()
                valid_dir = os.path.join(cache_dir, "autotrain", str(random_uuid))
                os.makedirs(valid_dir, exist_ok=True)
                self.valid_data.seek(0)
                content = self.valid_data.read()
                bytes_io = io.BytesIO(content)
                zip_ref = zipfile.ZipFile(bytes_io, "r")
                zip_ref.extractall(valid_dir)   
                macosx_dir = os.path.join(valid_dir, "__MACOSX")
                if os.path.exists(macosx_dir):
                    os.system(f"rm -rf {macosx_dir}")
                self._validate_audio_data(valid_dir)
                
        elif isinstance(self.train_data, str) and self.train_data.endswith('.zip') and os.path.isfile(self.train_data):
            cache_dir = os.environ.get("HF_HOME")
            if not cache_dir:
                cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
            
            random_uuid = uuid.uuid4()
            train_dir = os.path.join(cache_dir, "autotrain", str(random_uuid))
            train_dir = extract_zip_file(self.train_data, train_dir, task_type="audio")
            self._validate_audio_data(train_dir)
            
            if self.valid_data:
                random_uuid = uuid.uuid4()
                valid_dir = os.path.join(cache_dir, "autotrain", str(random_uuid))
                valid_dir = extract_zip_file(self.valid_data, valid_dir, task_type="audio")
                self._validate_audio_data(valid_dir)
                
        else:
            train_dir = self.train_data
            if self.valid_data:
                valid_dir = self.valid_data

        preprocessor = AudioClassificationPreprocessor(
            train_data=train_dir,
            valid_data=valid_dir,
            token=self.token,
            project_name=self.project_name,
            username=self.username,
            valid_split=self.percent_valid,
            local=self.local,
            audio_column="file_name",
            target_column="label",
        )
        return preprocessor.prepare()
    
    def _validate_audio_data(self, data_dir):
        """
        Validate that the extracted data contains audio files and metadata.jsonl
        """
        metadata_path = os.path.join(data_dir, "metadata.jsonl")
        if not os.path.exists(metadata_path):
            raise ValueError(f"metadata.jsonl not found in {data_dir}")
        
        audio_extensions = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
        audio_files = []
        for file in os.listdir(data_dir):
            if os.path.splitext(file)[1].lower() in audio_extensions:
                audio_files.append(file)
        
        if len(audio_files) < 1:
            raise ValueError(f"No audio files found in {data_dir}")
        
        import json
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if line.strip():
                        data = json.loads(line)
                        if 'file_name' not in data:
                            raise ValueError(f"Line {line_num}: 'file_name' field missing in metadata.jsonl")
                        if 'label' not in data:
                            raise ValueError(f"Line {line_num}: 'label' field missing in metadata.jsonl")
                        
                        audio_file = os.path.join(data_dir, data['file_name'])
                        if not os.path.exists(audio_file):
                            raise ValueError(f"Audio file {data['file_name']} referenced in metadata.jsonl not found")
                            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in metadata.jsonl: {e}")


@dataclass
class AutoTrainAudioSegmentationDataset:
    """
    A dataset class for AutoTrain audio segmentation tasks.

    Attributes:
        train_data (str): Path to the training data ZIP file.
        token (str): Authentication token.
        project_name (str): Name of the project.
        username (str): Username of the project owner.
        valid_data (Optional[str]): Path to the validation data ZIP file. Default is None.
        percent_valid (Optional[float]): Percentage of training data to be used for validation. Default is None.
        local (bool): Flag indicating if the data is local. Default is False.

    Methods:
        __str__() -> str:
            Returns a string representation of the dataset.

        __post_init__():
            Initializes the dataset and sets default values for validation data.

        prepare():
            Prepares the dataset for training by extracting and preprocessing the audio data.
    """

    train_data: str
    token: str
    project_name: str
    username: str
    valid_data: Optional[str] = None
    percent_valid: Optional[float] = None
    local: bool = False

    def __str__(self) -> str:
        info = f"Dataset: {self.project_name} ({self.task})\n"
        info += f"Train data: {self.train_data}\n"
        info += f"Valid data: {self.valid_data}\n"
        return info

    def __post_init__(self):
        self.task = "audio_segmentation"
        if not self.valid_data and self.percent_valid is None:
            self.percent_valid = 0.2
        elif self.valid_data and self.percent_valid is not None:
            raise ValueError("You can only specify one of valid_data or percent_valid")
        elif self.valid_data:
            self.percent_valid = 0.0

    def prepare(self):
        valid_dir = None
        if not isinstance(self.train_data, str):
            cache_dir = os.environ.get("HF_HOME")
            if not cache_dir:
                cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")

            random_uuid = uuid.uuid4()
            train_dir = os.path.join(cache_dir, "autotrain", str(random_uuid))
            os.makedirs(train_dir, exist_ok=True)
            self.train_data.seek(0)
            content = self.train_data.read()
            bytes_io = io.BytesIO(content)

            zip_ref = zipfile.ZipFile(bytes_io, "r")
            zip_ref.extractall(train_dir)
            macosx_dir = os.path.join(train_dir, "__MACOSX")
            if os.path.exists(macosx_dir):
                os.system(f"rm -rf {macosx_dir}")
            
            self._validate_audio_data(train_dir)
            
            if self.valid_data:
                random_uuid = uuid.uuid4()
                valid_dir = os.path.join(cache_dir, "autotrain", str(random_uuid))
                os.makedirs(valid_dir, exist_ok=True)
                self.valid_data.seek(0)
                content = self.valid_data.read()
                bytes_io = io.BytesIO(content)
                zip_ref = zipfile.ZipFile(bytes_io, "r")
                zip_ref.extractall(valid_dir)   
                macosx_dir = os.path.join(valid_dir, "__MACOSX")
                if os.path.exists(macosx_dir):
                    os.system(f"rm -rf {macosx_dir}")
                self._validate_audio_data(valid_dir)
                
        elif isinstance(self.train_data, str) and self.train_data.endswith('.zip') and os.path.isfile(self.train_data):
            cache_dir = os.environ.get("HF_HOME")
            if not cache_dir:
                cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
            
            random_uuid = uuid.uuid4()
            train_dir = os.path.join(cache_dir, "autotrain", str(random_uuid))
            train_dir = extract_zip_file(self.train_data, train_dir, task_type="audio")
            self._validate_audio_data(train_dir)
            
            if self.valid_data:
                random_uuid = uuid.uuid4()
                valid_dir = os.path.join(cache_dir, "autotrain", str(random_uuid))
                valid_dir = extract_zip_file(self.valid_data, valid_dir, task_type="audio")
                self._validate_audio_data(valid_dir)
                
        else:
            train_dir = self.train_data
            if self.valid_data:
                valid_dir = self.valid_data

        preprocessor = AudioSegmentationPreprocessor(
            train_data=train_dir,
            valid_data=valid_dir,
            token=self.token,
            project_name=self.project_name,
            username=self.username,
            valid_split=self.percent_valid,
            local=self.local,
            audio_column="file_name",
            segments_column="segments",
        )
        return preprocessor.prepare()
    
    def _validate_audio_data(self, data_dir):
        """
        Validate that the extracted data contains audio files and metadata.jsonl
        """
        metadata_path = os.path.join(data_dir, "metadata.jsonl")
        if not os.path.exists(metadata_path):
            raise ValueError(f"metadata.jsonl not found in {data_dir}")
        
        audio_extensions = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
        audio_files = []
        for file in os.listdir(data_dir):
            if os.path.splitext(file)[1].lower() in audio_extensions:
                audio_files.append(file)
        
        if len(audio_files) < 1:
            raise ValueError(f"No audio files found in {data_dir}")
        
        import json
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if line.strip():
                        data = json.loads(line)
                        if 'file_name' not in data:
                            raise ValueError(f"Line {line_num}: 'file_name' field missing in metadata.jsonl")
                        if 'segments' not in data:
                            raise ValueError(f"Line {line_num}: 'segments' field missing in metadata.jsonl")
                        
                        audio_file = os.path.join(data_dir, data['file_name'])
                        if not os.path.exists(audio_file):
                            raise ValueError(f"Audio file {data['file_name']} referenced in metadata.jsonl not found")
                            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in metadata.jsonl: {e}")


@dataclass
class AutoTrainAudioDetectionDataset:
    """
    AutoTrain Audio Detection Dataset
    
    Handles ZIP-based audio detection data with temporal event annotations.
    Similar to object detection but for audio events with precise timing.
    
    Expected format:
    - ZIP file containing audio files + metadata.jsonl
    - metadata.jsonl: {"file_name": "audio.wav", "events": [{"start": 4.23, "end": 4.27, "label": "car_crash"}]}
    """
    train_data: str
    token: str
    project_name: str
    username: str
    valid_data: Optional[str] = None
    percent_valid: Optional[float] = None
    local: bool = False

    def __str__(self) -> str:
        info = f"Dataset: {self.train_data}\n"
        if self.valid_data:
            info += f"Valid: {self.valid_data}\n"
        info += f"Task: Audio Detection\n"
        if self.username:
            info += f"Username: {self.username}\n"
        info += f"Project: {self.project_name}\n"
        info += f"Local: {self.local}"
        return info

    def __post_init__(self):
        if self.username is None:
            self.username = "autotrain-user"

        if self.valid_data is None and self.percent_valid is None:
            self.percent_valid = 0.2
        elif self.valid_data is not None and self.percent_valid is not None:
            raise ValueError("Please provide either valid_data or percent_valid, not both")

    def prepare(self):
        """
        Prepare the audio detection dataset from ZIP format.
        
        Expected ZIP structure:
        - audio_data.zip
          ├── audio1.wav
          ├── audio2.wav  
          └── metadata.jsonl
        
        metadata.jsonl format:
        {"file_name": "audio1.wav", "events": [{"start": 0.0, "end": 2.5, "label": "speech"}, {"start": 2.5, "end": 3.0, "label": "silence"}]}
        """
        from autotrain.preprocessor.audio import AudioDetectionPreprocessor
        
        if isinstance(self.train_data, str) and self.train_data.endswith('.zip'):
            processor = AudioDetectionPreprocessor(
                train_data=self.train_data,
                valid_data=self.valid_data,
                token=self.token,
                project_name=self.project_name,
                username=self.username,
                valid_split=self.percent_valid,
                seed=42,
                local=self.local,
                audio_column="file_name",
                events_column="events"
            )
        elif os.path.isdir(self.train_data):
            processor = AudioDetectionPreprocessor(
                train_data=self.train_data,
                valid_data=self.valid_data,
                token=self.token,
                project_name=self.project_name,
                username=self.username,
                valid_split=self.percent_valid,
                seed=42,
                local=self.local,
                audio_column="file_name",
                events_column="events"
            )
        else:
            processor = AudioDetectionPreprocessor(
                train_data=self.train_data,
                valid_data=self.valid_data,
                token=self.token,
                project_name=self.project_name,
                username=self.username,
                valid_split=self.percent_valid,
                seed=42,
                local=self.local,
                audio_column="file_name",
                events_column="events"
            )
        
        return processor.prepare()


@dataclass
class AutoTrainDataset:
    """
    AutoTrainDataset class for handling various types of datasets and preprocessing tasks.

    Attributes:
        train_data (List[str]): List of file paths or DataFrames for training data.
        task (str): The type of task to perform (e.g., "text_binary_classification").
        token (str): Authentication token.
        project_name (str): Name of the project.
        username (Optional[str]): Username of the project owner. Defaults to None.
        column_mapping (Optional[Dict[str, str]]): Mapping of column names. Defaults to None.
        valid_data (Optional[List[str]]): List of file paths or DataFrames for validation data. Defaults to None.
        percent_valid (Optional[float]): Percentage of training data to use for validation. Defaults to None.
        convert_to_class_label (Optional[bool]): Whether to convert labels to class labels. Defaults to False.
        local (bool): Whether the data is local. Defaults to False.
        ext (Optional[str]): File extension of the data files. Defaults to "csv".

    Methods:
        __str__(): Returns a string representation of the dataset.
        __post_init__(): Initializes validation data and preprocesses the data.
        _preprocess_data(): Preprocesses the training and validation data.
        num_samples(): Returns the total number of samples in the dataset.
        prepare(): Prepares the dataset for the specified task using the appropriate preprocessor.
    """

    train_data: List[str]
    task: str
    token: str
    project_name: str
    username: Optional[str] = None
    column_mapping: Optional[Dict[str, str]] = None
    valid_data: Optional[List[str]] = None
    percent_valid: Optional[float] = None
    convert_to_class_label: Optional[bool] = False
    local: bool = False
    ext: Optional[str] = "csv"

    def __str__(self) -> str:
        info = f"Dataset: {self.project_name} ({self.task})\n"
        info += f"Train data: {self.train_data}\n"
        info += f"Valid data: {self.valid_data}\n"
        info += f"Column mapping: {self.column_mapping}\n"
        return info

    def __post_init__(self):
        if self.valid_data is None:
            self.valid_data = []
        if not self.valid_data and self.percent_valid is None:
            self.percent_valid = 0.2
        elif self.valid_data and self.percent_valid is not None:
            raise ValueError("You can only specify one of valid_data or percent_valid")
        elif self.valid_data:
            self.percent_valid = 0.0

        self.train_df, self.valid_df = self._preprocess_data()

    def _preprocess_data(self):
        train_df = []
        for file in self.train_data:
            if isinstance(file, pd.DataFrame):
                train_df.append(file)
            else:
                if self.ext == "jsonl":
                    train_df.append(pd.read_json(file, lines=True))
                else:
                    train_df.append(pd.read_csv(file))
        if len(train_df) > 1:
            train_df = pd.concat(train_df)
        else:
            train_df = train_df[0]

        valid_df = None
        if len(self.valid_data) > 0:
            valid_df = []
            for file in self.valid_data:
                if isinstance(file, pd.DataFrame):
                    valid_df.append(file)
                else:
                    if self.ext == "jsonl":
                        valid_df.append(pd.read_json(file, lines=True))
                    else:
                        valid_df.append(pd.read_csv(file))
            if len(valid_df) > 1:
                valid_df = pd.concat(valid_df)
            else:
                valid_df = valid_df[0]
        return train_df, valid_df

    @property
    def num_samples(self):
        return len(self.train_df) + len(self.valid_df) if self.valid_df is not None else len(self.train_df)

    def prepare(self):
        if self.task == "text_binary_classification":
            text_column = self.column_mapping["text"]
            label_column = self.column_mapping["label"]
            preprocessor = TextBinaryClassificationPreprocessor(
                train_data=self.train_df,
                text_column=text_column,
                label_column=label_column,
                username=self.username,
                project_name=self.project_name,
                valid_data=self.valid_df,
                test_size=self.percent_valid,
                token=self.token,
                seed=42,
                convert_to_class_label=self.convert_to_class_label,
                local=self.local,
            )
            return preprocessor.prepare()

        elif self.task == "text_multi_class_classification":
            text_column = self.column_mapping["text"]
            label_column = self.column_mapping["label"]
            preprocessor = TextMultiClassClassificationPreprocessor(
                train_data=self.train_df,
                text_column=text_column,
                label_column=label_column,
                username=self.username,
                project_name=self.project_name,
                valid_data=self.valid_df,
                test_size=self.percent_valid,
                token=self.token,
                seed=42,
                convert_to_class_label=self.convert_to_class_label,
                local=self.local,
            )
            return preprocessor.prepare()

        elif self.task == "text_token_classification":
            text_column = self.column_mapping["text"]
            label_column = self.column_mapping["label"]
            preprocessor = TextTokenClassificationPreprocessor(
                train_data=self.train_df,
                text_column=text_column,
                label_column=label_column,
                username=self.username,
                project_name=self.project_name,
                valid_data=self.valid_df,
                test_size=self.percent_valid,
                token=self.token,
                seed=42,
                local=self.local,
                convert_to_class_label=self.convert_to_class_label,
            )
            return preprocessor.prepare()

        elif self.task == "text_single_column_regression":
            text_column = self.column_mapping["text"]
            label_column = self.column_mapping["label"]
            preprocessor = TextSingleColumnRegressionPreprocessor(
                train_data=self.train_df,
                text_column=text_column,
                label_column=label_column,
                username=self.username,
                project_name=self.project_name,
                valid_data=self.valid_df,
                test_size=self.percent_valid,
                token=self.token,
                seed=42,
                local=self.local,
            )
            return preprocessor.prepare()

        elif self.task == "seq2seq":
            text_column = self.column_mapping["text"]
            label_column = self.column_mapping["label"]
            preprocessor = Seq2SeqPreprocessor(
                train_data=self.train_df,
                text_column=text_column,
                label_column=label_column,
                username=self.username,
                project_name=self.project_name,
                valid_data=self.valid_df,
                test_size=self.percent_valid,
                token=self.token,
                seed=42,
                local=self.local,
            )
            return preprocessor.prepare()

        elif self.task == "lm_training":
            text_column = self.column_mapping["text"]
            prompt_column = self.column_mapping.get("prompt")
            rejected_text_column = self.column_mapping.get("rejected_text")
            preprocessor = LLMPreprocessor(
                train_data=self.train_df,
                text_column=text_column,
                prompt_column=prompt_column,
                rejected_text_column=rejected_text_column,
                username=self.username,
                project_name=self.project_name,
                valid_data=self.valid_df,
                test_size=self.percent_valid,
                token=self.token,
                seed=42,
                local=self.local,
            )
            return preprocessor.prepare()

        elif self.task == "sentence_transformers":
            sentence1_column = self.column_mapping["sentence1"]
            sentence2_column = self.column_mapping["sentence2"]
            sentence3_column = self.column_mapping.get("sentence3")
            target_column = self.column_mapping.get("target")

            preprocessor = SentenceTransformersPreprocessor(
                train_data=self.train_df,
                username=self.username,
                project_name=self.project_name,
                valid_data=self.valid_df,
                test_size=self.percent_valid,
                token=self.token,
                seed=42,
                local=self.local,
                sentence1_column=sentence1_column,
                sentence2_column=sentence2_column,
                sentence3_column=sentence3_column,
                target_column=target_column,
                convert_to_class_label=self.convert_to_class_label,
            )
            return preprocessor.prepare()

        elif self.task == "text_extractive_question_answering":
            text_column = self.column_mapping["text"]
            question_column = self.column_mapping["question"]
            answer_column = self.column_mapping["answer"]
            preprocessor = TextExtractiveQuestionAnsweringPreprocessor(
                train_data=self.train_df,
                text_column=text_column,
                question_column=question_column,
                answer_column=answer_column,
                username=self.username,
                project_name=self.project_name,
                valid_data=self.valid_df,
                test_size=self.percent_valid,
                token=self.token,
                seed=42,
                local=self.local,
            )
            return preprocessor.prepare()

        elif self.task == "tabular_binary_classification":
            id_column = self.column_mapping["id"]
            label_column = self.column_mapping["label"][0]
            if len(id_column.strip()) == 0:
                id_column = None
            preprocessor = TabularBinaryClassificationPreprocessor(
                train_data=self.train_df,
                id_column=id_column,
                label_column=label_column,
                username=self.username,
                project_name=self.project_name,
                valid_data=self.valid_df,
                test_size=self.percent_valid,
                token=self.token,
                seed=42,
                local=self.local,
            )
            return preprocessor.prepare()
        elif self.task == "tabular_multi_class_classification":
            id_column = self.column_mapping["id"]
            label_column = self.column_mapping["label"][0]
            if len(id_column.strip()) == 0:
                id_column = None
            preprocessor = TabularMultiClassClassificationPreprocessor(
                train_data=self.train_df,
                id_column=id_column,
                label_column=label_column,
                username=self.username,
                project_name=self.project_name,
                valid_data=self.valid_df,
                test_size=self.percent_valid,
                token=self.token,
                seed=42,
                local=self.local,
            )
            return preprocessor.prepare()
        elif self.task == "tabular_single_column_regression":
            id_column = self.column_mapping["id"]
            label_column = self.column_mapping["label"][0]
            if len(id_column.strip()) == 0:
                id_column = None
            preprocessor = TabularSingleColumnRegressionPreprocessor(
                train_data=self.train_df,
                id_column=id_column,
                label_column=label_column,
                username=self.username,
                project_name=self.project_name,
                valid_data=self.valid_df,
                test_size=self.percent_valid,
                token=self.token,
                seed=42,
                local=self.local,
            )
            return preprocessor.prepare()
        elif self.task == "tabular_multi_column_regression":
            id_column = self.column_mapping["id"]
            label_column = self.column_mapping["label"]
            if len(id_column.strip()) == 0:
                id_column = None
            preprocessor = TabularMultiColumnRegressionPreprocessor(
                train_data=self.train_df,
                id_column=id_column,
                label_column=label_column,
                username=self.username,
                project_name=self.project_name,
                valid_data=self.valid_df,
                test_size=self.percent_valid,
                token=self.token,
                seed=42,
                local=self.local,
            )
            return preprocessor.prepare()
        elif self.task == "tabular_multi_label_classification":
            id_column = self.column_mapping["id"]
            label_column = self.column_mapping["label"]
            if len(id_column.strip()) == 0:
                id_column = None
            preprocessor = TabularMultiLabelClassificationPreprocessor(
                train_data=self.train_df,
                id_column=id_column,
                label_column=label_column,
                username=self.username,
                project_name=self.project_name,
                valid_data=self.valid_df,
                test_size=self.percent_valid,
                token=self.token,
                seed=42,
                local=self.local,
            )
            return preprocessor.prepare()
        elif self.task in ["audio_binary_classification", "audio_multi_class_classification"]:
            audio_column = self.column_mapping["audio"]
            label_column = self.column_mapping["label"]
            preprocessor = AudioClassificationPreprocessor(
                train_data=self.train_df,
                valid_data=self.valid_df,
                token=self.token,
                project_name=self.project_name,
                username=self.username,
                valid_split=self.percent_valid,
                local=self.local,
                audio_column=audio_column,
                target_column=label_column,
            )
            return preprocessor.prepare()
        elif self.task == "audio_segmentation":
            audio_column = self.column_mapping["audio"]
            segments_column = self.column_mapping["label"]
            preprocessor = AudioSegmentationPreprocessor(
                train_data=self.train_df,
                valid_data=self.valid_df,
                token=self.token,
                project_name=self.project_name,
                username=self.username,
                valid_split=self.percent_valid,
                local=self.local,
                audio_column=audio_column,
                segments_column=segments_column,
            )
            return preprocessor.prepare()
        elif self.task == "audio_detection":
            audio_column = self.column_mapping["audio"]
            events_column = self.column_mapping["events"]
            preprocessor = AudioDetectionPreprocessor(
                train_data=self.train_df,
                valid_data=self.valid_df,
                token=self.token,
                project_name=self.project_name,
                username=self.username,
                valid_split=self.percent_valid,
                local=self.local,
                audio_column=audio_column,
                events_column=events_column,
            )
            return preprocessor.prepare()
        else:
            raise ValueError(f"Task {self.task} not supported")
