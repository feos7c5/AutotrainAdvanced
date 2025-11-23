import os
import shutil
import uuid
from dataclasses import dataclass
from typing import Optional

import pandas as pd
from datasets import ClassLabel, Features, Image, Sequence, Value, load_dataset
from sklearn.model_selection import train_test_split


ALLOWED_EXTENSIONS = ("jpeg", "png", "jpg", "JPG", "JPEG", "PNG")


@dataclass
class ImageClassificationPreprocessor:
    """
    A class used to preprocess image data for classification tasks.

    Attributes
    ----------
    train_data : str
        Path to the training data directory.
    username : str
        Username for the Hugging Face Hub.
    project_name : str
        Name of the project.
    token : str
        Authentication token for the Hugging Face Hub.
    valid_data : Optional[str], optional
        Path to the validation data directory, by default None.
    test_size : Optional[float], optional
        Proportion of the dataset to include in the validation split, by default 0.2.
    seed : Optional[int], optional
        Random seed for reproducibility, by default 42.
    local : Optional[bool], optional
        Whether to save the dataset locally or push to the Hugging Face Hub, by default False.

    Methods
    -------
    __post_init__():
        Validates the structure and contents of the training and validation data directories.
    split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        Splits the dataframe into training and validation sets.
    prepare() -> str:
        Prepares the dataset for training and either saves it locally or pushes it to the Hugging Face Hub.
    """

    train_data: str
    username: str
    project_name: str
    token: str
    valid_data: Optional[str] = None
    test_size: Optional[float] = 0.2
    seed: Optional[int] = 42
    local: Optional[bool] = False

    def __post_init__(self):
        # Check if train data path exists
        if not os.path.exists(self.train_data):
            raise ValueError(f"{self.train_data} does not exist.")

        # Check if train data path contains at least 2 folders
        subfolders = [f.path for f in os.scandir(self.train_data) if f.is_dir()]
        # list subfolders
        if len(subfolders) < 2:
            raise ValueError(f"{self.train_data} should contain at least 2 subfolders.")

        # Check if each subfolder contains at least 2 image files in jpeg, png or jpg format only
        for subfolder in subfolders:
            image_files = [f for f in os.listdir(subfolder) if f.endswith(ALLOWED_EXTENSIONS)]
            if len(image_files) < 2:
                raise ValueError(f"{subfolder} should contain at least 2 jpeg, png or jpg files.")
            # Check if there are no other files except image files in the subfolder
            if len(image_files) != len(os.listdir(subfolder)):
                raise ValueError(f"{subfolder} should not contain any other files except image files.")

            # Check if there are no subfolders inside subfolders
            subfolders_in_subfolder = [f.path for f in os.scandir(subfolder) if f.is_dir()]
            if len(subfolders_in_subfolder) > 0:
                raise ValueError(f"{subfolder} should not contain any subfolders.")

        if self.valid_data:
            # Check if valid data path exists
            if not os.path.exists(self.valid_data):
                raise ValueError(f"{self.valid_data} does not exist.")

            # Check if valid data path contains at least 2 folders
            subfolders = [f.path for f in os.scandir(self.valid_data) if f.is_dir()]

            # make sure that the subfolders in train and valid data are the same
            train_subfolders = set(os.path.basename(f.path) for f in os.scandir(self.train_data) if f.is_dir())
            valid_subfolders = set(os.path.basename(f.path) for f in os.scandir(self.valid_data) if f.is_dir())
            if train_subfolders != valid_subfolders:
                raise ValueError(f"{self.valid_data} should have the same subfolders as {self.train_data}.")

            if len(subfolders) < 2:
                raise ValueError(f"{self.valid_data} should contain at least 2 subfolders.")

            # Check if each subfolder contains at least 2 image files in jpeg, png or jpg format only
            for subfolder in subfolders:
                image_files = [f for f in os.listdir(subfolder) if f.endswith(ALLOWED_EXTENSIONS)]
                if len(image_files) < 2:
                    raise ValueError(f"{subfolder} should contain at least 2 jpeg, png or jpg files.")

                # Check if there are no other files except image files in the subfolder
                if len(image_files) != len(os.listdir(subfolder)):
                    raise ValueError(f"{subfolder} should not contain any other files except image files.")

                # Check if there are no subfolders inside subfolders
                subfolders_in_subfolder = [f.path for f in os.scandir(subfolder) if f.is_dir()]
                if len(subfolders_in_subfolder) > 0:
                    raise ValueError(f"{subfolder} should not contain any subfolders.")

    def split(self, df):
        train_df, valid_df = train_test_split(
            df,
            test_size=self.test_size,
            random_state=self.seed,
            stratify=df["subfolder"],
        )
        train_df = train_df.reset_index(drop=True)
        valid_df = valid_df.reset_index(drop=True)
        return train_df, valid_df

    def prepare(self):
        random_uuid = uuid.uuid4()
        cache_dir = os.environ.get("HF_HOME")
        if not cache_dir:
            cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
        data_dir = os.path.join(cache_dir, "autotrain", str(random_uuid))

        if self.valid_data:
            shutil.copytree(self.train_data, os.path.join(data_dir, "train"))
            shutil.copytree(self.valid_data, os.path.join(data_dir, "validation"))

            dataset = load_dataset("imagefolder", data_dir=data_dir)
            dataset = dataset.rename_columns({"image": "autotrain_image", "label": "autotrain_label"})
            if self.local:
                dataset.save_to_disk(f"{self.project_name}/autotrain-data")
            else:
                dataset.push_to_hub(
                    f"{self.username}/autotrain-data-{self.project_name}",
                    private=True,
                    token=self.token,
                )

        else:
            subfolders = [f.path for f in os.scandir(self.train_data) if f.is_dir()]

            image_filenames = []
            subfolder_names = []

            for subfolder in subfolders:
                for filename in os.listdir(subfolder):
                    if filename.endswith(ALLOWED_EXTENSIONS):
                        image_filenames.append(filename)
                        subfolder_names.append(os.path.basename(subfolder))

            df = pd.DataFrame({"image_filename": image_filenames, "subfolder": subfolder_names})
            train_df, valid_df = self.split(df)

            for row in train_df.itertuples():
                os.makedirs(os.path.join(data_dir, "train", row.subfolder), exist_ok=True)
                shutil.copy(
                    os.path.join(self.train_data, row.subfolder, row.image_filename),
                    os.path.join(data_dir, "train", row.subfolder, row.image_filename),
                )

            for row in valid_df.itertuples():
                os.makedirs(os.path.join(data_dir, "validation", row.subfolder), exist_ok=True)
                shutil.copy(
                    os.path.join(self.train_data, row.subfolder, row.image_filename),
                    os.path.join(data_dir, "validation", row.subfolder, row.image_filename),
                )

            dataset = load_dataset("imagefolder", data_dir=data_dir)
            dataset = dataset.rename_columns({"image": "autotrain_image", "label": "autotrain_label"})
            if self.local:
                dataset.save_to_disk(f"{self.project_name}/autotrain-data")
            else:
                dataset.push_to_hub(
                    f"{self.username}/autotrain-data-{self.project_name}",
                    private=True,
                    token=self.token,
                )

        if self.local:
            return f"{self.project_name}/autotrain-data"
        return f"{self.username}/autotrain-data-{self.project_name}"


@dataclass
class ObjectDetectionPreprocessor:
    """
    A class to preprocess data for object detection tasks.

    Attributes:
    -----------
    train_data : str
        Path to the training data directory.
    username : str
        Username for the Hugging Face Hub.
    project_name : str
        Name of the project.
    token : str
        Authentication token for the Hugging Face Hub.
    valid_data : Optional[str], default=None
        Path to the validation data directory.
    test_size : Optional[float], default=0.2
        Proportion of the dataset to include in the validation split.
    seed : Optional[int], default=42
        Random seed for reproducibility.
    local : Optional[bool], default=False
        Whether to save the dataset locally or push to the Hugging Face Hub.

    Methods:
    --------
    _process_metadata(data_path):
        Processes the metadata.jsonl file and extracts required columns and categories.
    __post_init__():
        Validates the existence and content of the training and validation data directories.
    split(df):
        Splits the dataframe into training and validation sets.
    prepare():
        Prepares the dataset for training by processing metadata, splitting data, and saving or pushing the dataset.
    """

    train_data: str
    username: str
    project_name: str
    token: str
    valid_data: Optional[str] = None
    test_size: Optional[float] = 0.2
    seed: Optional[int] = 42
    local: Optional[bool] = False

    @staticmethod
    def _process_metadata(data_path):
        metadata = pd.read_json(os.path.join(data_path, "metadata.jsonl"), lines=True)
        # make sure that the metadata.jsonl file contains the required columns: file_name, objects
        if "file_name" not in metadata.columns or "objects" not in metadata.columns:
            raise ValueError(f"{data_path}/metadata.jsonl should contain 'file_name' and 'objects' columns.")

        # keep only file_name and objects columns
        metadata = metadata[["file_name", "objects"]]
        # inside metadata objects column, values should be bbox, area and category
        # if area does not exist, it should be created by multiplying bbox width and height
        categories = []
        for _, row in metadata.iterrows():
            obj = row["objects"]
            if "bbox" not in obj or "category" not in obj:
                raise ValueError(f"{data_path}/metadata.jsonl should contain 'bbox' and 'category' keys in 'objects'.")
            # keep only bbox, area and category keys
            obj = {k: obj[k] for k in ["bbox", "category"]}
            categories.extend(obj["category"])

        categories = set(categories)

        return metadata, categories

    def __post_init__(self):
        # Check if train data path exists
        if not os.path.exists(self.train_data):
            raise ValueError(f"{self.train_data} does not exist.")

        # check if self.train_data contains at least 5 image files in jpeg, png or jpg format only
        train_image_files = [f for f in os.listdir(self.train_data) if f.endswith(ALLOWED_EXTENSIONS)]
        if len(train_image_files) < 5:
            raise ValueError(f"{self.train_data} should contain at least 5 jpeg, png or jpg files.")

        # check if self.train_data contains a metadata.jsonl file
        if "metadata.jsonl" not in os.listdir(self.train_data):
            raise ValueError(f"{self.train_data} should contain a metadata.jsonl file.")

        # Check if valid data path exists
        if self.valid_data:
            if not os.path.exists(self.valid_data):
                raise ValueError(f"{self.valid_data} does not exist.")

            # check if self.valid_data contains at least 5 image files in jpeg, png or jpg format only
            valid_image_files = [f for f in os.listdir(self.valid_data) if f.endswith(ALLOWED_EXTENSIONS)]
            if len(valid_image_files) < 5:
                raise ValueError(f"{self.valid_data} should contain at least 5 jpeg, png or jpg files.")

            # check if self.valid_data contains a metadata.jsonl file
            if "metadata.jsonl" not in os.listdir(self.valid_data):
                raise ValueError(f"{self.valid_data} should contain a metadata.jsonl file.")

    def split(self, df):
        train_df, valid_df = train_test_split(
            df,
            test_size=self.test_size,
            random_state=self.seed,
        )
        train_df = train_df.reset_index(drop=True)
        valid_df = valid_df.reset_index(drop=True)
        return train_df, valid_df

    def prepare(self):
        random_uuid = uuid.uuid4()
        cache_dir = os.environ.get("HF_HOME")
        if not cache_dir:
            cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
        data_dir = os.path.join(cache_dir, "autotrain", str(random_uuid))

        if self.valid_data:
            shutil.copytree(self.train_data, os.path.join(data_dir, "train"))
            shutil.copytree(self.valid_data, os.path.join(data_dir, "validation"))

            train_metadata, train_categories = self._process_metadata(os.path.join(data_dir, "train"))
            valid_metadata, valid_categories = self._process_metadata(os.path.join(data_dir, "validation"))

            train_metadata.to_json(os.path.join(data_dir, "train", "metadata.jsonl"), orient="records", lines=True)
            valid_metadata.to_json(
                os.path.join(data_dir, "validation", "metadata.jsonl"), orient="records", lines=True
            )

            all_categories = train_categories.union(valid_categories)

            features = Features(
                {
                    "image": Image(),
                    "objects": Sequence(
                        {
                            "bbox": Sequence(Value("float32"), length=4),
                            "category": ClassLabel(names=list(all_categories)),
                        }
                    ),
                }
            )

            dataset = load_dataset("imagefolder", data_dir=data_dir, features=features)
            dataset = dataset.rename_columns(
                {
                    "image": "autotrain_image",
                    "objects": "autotrain_objects",
                }
            )

            if self.local:
                dataset.save_to_disk(f"{self.project_name}/autotrain-data")
            else:
                dataset.push_to_hub(
                    f"{self.username}/autotrain-data-{self.project_name}",
                    private=True,
                    token=self.token,
                )
        else:
            metadata = pd.read_json(os.path.join(self.train_data, "metadata.jsonl"), lines=True)
            train_df, valid_df = self.split(metadata)

            # create train and validation folders
            os.makedirs(os.path.join(data_dir, "train"), exist_ok=True)
            os.makedirs(os.path.join(data_dir, "validation"), exist_ok=True)

            # move images to train and validation folders
            for row in train_df.iterrows():
                shutil.copy(
                    os.path.join(self.train_data, row[1]["file_name"]),
                    os.path.join(data_dir, "train", row[1]["file_name"]),
                )

            for row in valid_df.iterrows():
                shutil.copy(
                    os.path.join(self.train_data, row[1]["file_name"]),
                    os.path.join(data_dir, "validation", row[1]["file_name"]),
                )

            # save metadata.jsonl file to train and validation folders
            train_df.to_json(os.path.join(data_dir, "train", "metadata.jsonl"), orient="records", lines=True)
            valid_df.to_json(os.path.join(data_dir, "validation", "metadata.jsonl"), orient="records", lines=True)

            train_metadata, train_categories = self._process_metadata(os.path.join(data_dir, "train"))
            valid_metadata, valid_categories = self._process_metadata(os.path.join(data_dir, "validation"))

            train_metadata.to_json(os.path.join(data_dir, "train", "metadata.jsonl"), orient="records", lines=True)
            valid_metadata.to_json(
                os.path.join(data_dir, "validation", "metadata.jsonl"), orient="records", lines=True
            )

            all_categories = train_categories.union(valid_categories)

            features = Features(
                {
                    "image": Image(),
                    "objects": Sequence(
                        {
                            "bbox": Sequence(Value("float32"), length=4),
                            "category": ClassLabel(names=list(all_categories)),
                        }
                    ),
                }
            )

            dataset = load_dataset("imagefolder", data_dir=data_dir, features=features)
            dataset = dataset.rename_columns(
                {
                    "image": "autotrain_image",
                    "objects": "autotrain_objects",
                }
            )

            if self.local:
                dataset.save_to_disk(f"{self.project_name}/autotrain-data")
            else:
                dataset.push_to_hub(
                    f"{self.username}/autotrain-data-{self.project_name}",
                    private=True,
                    token=self.token,
                )

        if self.local:
            return f"{self.project_name}/autotrain-data"
        return f"{self.username}/autotrain-data-{self.project_name}"


@dataclass
class ImageRegressionPreprocessor:
    train_data: str
    username: str
    project_name: str
    token: str
    valid_data: Optional[str] = None
    test_size: Optional[float] = 0.2
    seed: Optional[int] = 42
    local: Optional[bool] = False

    @staticmethod
    def _process_metadata(data_path):
        metadata = pd.read_json(os.path.join(data_path, "metadata.jsonl"), lines=True)
        # make sure that the metadata.jsonl file contains the required columns: file_name, target
        if "file_name" not in metadata.columns or "target" not in metadata.columns:
            raise ValueError(f"{data_path}/metadata.jsonl should contain 'file_name' and 'target' columns.")

        # keep only file_name and target columns
        metadata = metadata[["file_name", "target"]]
        return metadata

    def __post_init__(self):
        # Check if train data path exists
        if not os.path.exists(self.train_data):
            raise ValueError(f"{self.train_data} does not exist.")

        # check if self.train_data contains at least 5 image files in jpeg, png or jpg format only
        train_image_files = [f for f in os.listdir(self.train_data) if f.endswith(ALLOWED_EXTENSIONS)]
        if len(train_image_files) < 5:
            raise ValueError(f"{self.train_data} should contain at least 5 jpeg, png or jpg files.")

        # check if self.train_data contains a metadata.jsonl file
        if "metadata.jsonl" not in os.listdir(self.train_data):
            raise ValueError(f"{self.train_data} should contain a metadata.jsonl file.")

        # Check if valid data path exists
        if self.valid_data:
            if not os.path.exists(self.valid_data):
                raise ValueError(f"{self.valid_data} does not exist.")

            # check if self.valid_data contains at least 5 image files in jpeg, png or jpg format only
            valid_image_files = [f for f in os.listdir(self.valid_data) if f.endswith(ALLOWED_EXTENSIONS)]
            if len(valid_image_files) < 5:
                raise ValueError(f"{self.valid_data} should contain at least 5 jpeg, png or jpg files.")

            # check if self.valid_data contains a metadata.jsonl file
            if "metadata.jsonl" not in os.listdir(self.valid_data):
                raise ValueError(f"{self.valid_data} should contain a metadata.jsonl file.")

    def split(self, df):
        train_df, valid_df = train_test_split(
            df,
            test_size=self.test_size,
            random_state=self.seed,
        )
        train_df = train_df.reset_index(drop=True)
        valid_df = valid_df.reset_index(drop=True)
        return train_df, valid_df

    def prepare(self):
        random_uuid = uuid.uuid4()
        cache_dir = os.environ.get("HF_HOME")
        if not cache_dir:
            cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
        data_dir = os.path.join(cache_dir, "autotrain", str(random_uuid))

        if self.valid_data:
            shutil.copytree(self.train_data, os.path.join(data_dir, "train"))
            shutil.copytree(self.valid_data, os.path.join(data_dir, "validation"))

            train_metadata = self._process_metadata(os.path.join(data_dir, "train"))
            valid_metadata = self._process_metadata(os.path.join(data_dir, "validation"))

            train_metadata.to_json(os.path.join(data_dir, "train", "metadata.jsonl"), orient="records", lines=True)
            valid_metadata.to_json(
                os.path.join(data_dir, "validation", "metadata.jsonl"), orient="records", lines=True
            )

            dataset = load_dataset("imagefolder", data_dir=data_dir)
            dataset = dataset.rename_columns(
                {
                    "image": "autotrain_image",
                    "target": "autotrain_label",
                }
            )

            if self.local:
                dataset.save_to_disk(f"{self.project_name}/autotrain-data")
            else:
                dataset.push_to_hub(
                    f"{self.username}/autotrain-data-{self.project_name}",
                    private=True,
                    token=self.token,
                )
        else:
            metadata = pd.read_json(os.path.join(self.train_data, "metadata.jsonl"), lines=True)
            train_df, valid_df = self.split(metadata)

            # create train and validation folders
            os.makedirs(os.path.join(data_dir, "train"), exist_ok=True)
            os.makedirs(os.path.join(data_dir, "validation"), exist_ok=True)

            # move images to train and validation folders
            for row in train_df.iterrows():
                shutil.copy(
                    os.path.join(self.train_data, row[1]["file_name"]),
                    os.path.join(data_dir, "train", row[1]["file_name"]),
                )

            for row in valid_df.iterrows():
                shutil.copy(
                    os.path.join(self.train_data, row[1]["file_name"]),
                    os.path.join(data_dir, "validation", row[1]["file_name"]),
                )

            # save metadata.jsonl file to train and validation folders
            train_df.to_json(os.path.join(data_dir, "train", "metadata.jsonl"), orient="records", lines=True)
            valid_df.to_json(os.path.join(data_dir, "validation", "metadata.jsonl"), orient="records", lines=True)

            train_metadata = self._process_metadata(os.path.join(data_dir, "train"))
            valid_metadata = self._process_metadata(os.path.join(data_dir, "validation"))

            train_metadata.to_json(os.path.join(data_dir, "train", "metadata.jsonl"), orient="records", lines=True)
            valid_metadata.to_json(
                os.path.join(data_dir, "validation", "metadata.jsonl"), orient="records", lines=True
            )

            dataset = load_dataset("imagefolder", data_dir=data_dir)
            dataset = dataset.rename_columns(
                {
                    "image": "autotrain_image",
                    "target": "autotrain_label",
                }
            )

            if self.local:
                dataset.save_to_disk(f"{self.project_name}/autotrain-data")
            else:
                dataset.push_to_hub(
                    f"{self.username}/autotrain-data-{self.project_name}",
                    private=True,
                    token=self.token,
                )

        if self.local:
            return f"{self.project_name}/autotrain-data"
        return f"{self.username}/autotrain-data-{self.project_name}"


@dataclass
class ImageSemanticSegmentationPreprocessor:
    """
    A class used to preprocess image data for semantic segmentation tasks.
    
    Supports the standard semantic segmentation format used by CVAT, Supervisely, Pascal VOC:
    - images/ folder with RGB images (JPG/PNG)
    - masks/ or annotations/ folder with grayscale PNG masks (pixel values = class IDs)
    - Optional classes.txt or labelmap.txt with class names (one per line)
    
    Expected input structure (directory):
    ```
    dataset/
    ├── images/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    ├── masks/ (or annotations/)
    │   ├── image1.png
    │   ├── image2.png  
    │   └── ...
    └── classes.txt (optional)
    ```

    Attributes
    ----------
    train_data : str
        Path to the training data directory.
    username : str
        Username for the Hugging Face Hub.
    project_name : str
        Name of the project.
    token : str
        Authentication token for the Hugging Face Hub.
    valid_data : Optional[str], optional
        Path to the validation data directory, by default None.
    test_size : Optional[float], optional
        Proportion of the dataset to include in the validation split, by default 0.2.
    seed : Optional[int], optional
        Random seed for reproducibility, by default 42.
    local : Optional[bool], optional
        Whether to save the dataset locally or push to the Hugging Face Hub, by default False.

    Methods
    -------
    __post_init__():
        Validates the structure and contents of the training and validation data.
    prepare() -> str:
        Prepares the dataset for training and either saves it locally or pushes it to the Hugging Face Hub.
    """

    train_data: str
    username: str
    project_name: str
    token: str
    valid_data: Optional[str] = None
    test_size: Optional[float] = 0.2
    seed: Optional[int] = 42
    local: Optional[bool] = False

    @staticmethod
    def _find_images_and_masks_dirs(data_dir):
        """
        Find images and masks directories.
        Supports common naming conventions: images/masks, img/ann, images/annotations, etc.
        """
        # Common directory name patterns
        image_dir_names = ['images', 'img', 'image', 'imgs']
        mask_dir_names = ['masks', 'mask', 'annotations', 'ann', 'segmentations', 'labels']
        
        images_dir = None
        masks_dir = None
        
        # Look for subdirectories
        subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        
        # Find images directory
        for subdir in subdirs:
            if subdir.lower() in image_dir_names:
                images_dir = os.path.join(data_dir, subdir)
                break
        
        # Find masks directory
        for subdir in subdirs:
            if subdir.lower() in mask_dir_names:
                masks_dir = os.path.join(data_dir, subdir)
                break
        
        # If no subdirectories found, assume flat structure
        if images_dir is None or masks_dir is None:
            # Check if we have images and masks in the same directory
            files = os.listdir(data_dir)
            image_files = [f for f in files if f.lower().endswith(ALLOWED_EXTENSIONS)]
            mask_files = [f for f in files if f.lower().endswith('.png')]
            
            if len(image_files) > 0 and len(mask_files) > 0:
                return data_dir, data_dir  # Same directory for both
        
        return images_dir, masks_dir

    @staticmethod
    def _find_classes_file(data_dir):
        """Find the path to classes.txt or similar file if available."""
        class_file_names = ['classes.txt', 'labelmap.txt', 'labels.txt']
        
        for class_file in class_file_names:
            class_path = os.path.join(data_dir, class_file)
            if os.path.exists(class_path):
                return class_path
        
        return None

    @staticmethod
    def _load_classes(data_dir):
        """Load class names from classes.txt or labelmap.txt if available."""
        class_file_names = ['classes.txt', 'labelmap.txt', 'labels.txt']
        
        for class_file in class_file_names:
            class_path = os.path.join(data_dir, class_file)
            if os.path.exists(class_path):
                with open(class_path, 'r') as f:
                    classes = [line.strip() for line in f.readlines() if line.strip()]
                return classes
        
        return None

    @staticmethod
    def _get_unique_mask_values(mask_dir):
        """Get unique pixel values from all mask files to determine number of classes."""
        import numpy as np
        from PIL import Image as PILImage
        
        unique_values = set()
        mask_files = [f for f in os.listdir(mask_dir) if f.lower().endswith('.png')]
        
        for mask_file in mask_files[:10]:  # Sample first 10 masks to determine classes
            mask_path = os.path.join(mask_dir, mask_file)
            mask = PILImage.open(mask_path)
            mask_array = np.array(mask)
            unique_values.update(np.unique(mask_array).tolist())
        
        return sorted(list(unique_values))

    @staticmethod
    def _validate_dataset_structure(images_dir, masks_dir):
        """Validate that the dataset has proper structure."""
        if not os.path.exists(images_dir):
            raise ValueError(f"Images directory not found: {images_dir}")
        
        if not os.path.exists(masks_dir):
            raise ValueError(f"Masks directory not found: {masks_dir}")
        
        # Get image and mask files
        image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(ALLOWED_EXTENSIONS)]
        mask_files = [f for f in os.listdir(masks_dir) if f.lower().endswith('.png')]
        
        if len(image_files) == 0:
            raise ValueError(f"No image files found in {images_dir}")
        
        if len(mask_files) == 0:
            raise ValueError(f"No mask files found in {masks_dir}")
        
        # Check if we have corresponding masks for images
        image_basenames = {os.path.splitext(f)[0] for f in image_files}
        mask_basenames = {os.path.splitext(f)[0] for f in mask_files}
        
        missing_masks = image_basenames - mask_basenames
        if missing_masks:
            raise ValueError(f"Missing masks for images: {list(missing_masks)[:5]}...")
        
        return image_files, mask_files

    def __post_init__(self):
        if not os.path.exists(self.train_data):
            raise ValueError(f"{self.train_data} does not exist.")
        
        # Find and validate training data structure
        train_images_dir, train_masks_dir = self._find_images_and_masks_dirs(self.train_data)
        self._validate_dataset_structure(train_images_dir, train_masks_dir)
        
        # Validate validation data if provided
        if self.valid_data:
            if not os.path.exists(self.valid_data):
                raise ValueError(f"{self.valid_data} does not exist.")
            
            valid_images_dir, valid_masks_dir = self._find_images_and_masks_dirs(self.valid_data)
            self._validate_dataset_structure(valid_images_dir, valid_masks_dir)

    def split(self, df):
        """Split dataframe into train and validation sets."""
        train_df, valid_df = train_test_split(
            df,
            test_size=self.test_size,
            random_state=self.seed,
        )
        train_df = train_df.reset_index(drop=True)
        valid_df = valid_df.reset_index(drop=True)
        return train_df, valid_df

    def prepare(self):
        """
        Prepare the semantic segmentation dataset following the standard format.
        Creates a Hugging Face dataset with images and segmentation masks.
        """
        from datasets import Dataset, DatasetDict, Features, Image, ClassLabel
        import numpy as np
        from PIL import Image as PILImage
        
        random_uuid = uuid.uuid4()
        cache_dir = os.environ.get("HF_HOME")
        if not cache_dir:
            cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
        data_dir = os.path.join(cache_dir, "autotrain", str(random_uuid))
        
        # Process training data
        train_images_dir, train_masks_dir = self._find_images_and_masks_dirs(self.train_data)
        train_image_files, train_mask_files = self._validate_dataset_structure(train_images_dir, train_masks_dir)
        
        # Load class names if available
        classes = self._load_classes(self.train_data)
        if classes is None:
            # Determine classes from unique mask values
            unique_values = self._get_unique_mask_values(train_masks_dir)
            classes = [f"class_{i}" for i in unique_values]
        
        num_classes = len(classes)
        print(f"Found {num_classes} classes: {classes}")
        
        if self.valid_data:
            # Use provided validation data
            shutil.copytree(self.train_data, os.path.join(data_dir, "train"))
            shutil.copytree(self.valid_data, os.path.join(data_dir, "validation"))
        else:
            # Create train/validation split
            # Create DataFrame with image-mask pairs
            image_mask_pairs = []
            for img_file in train_image_files:
                img_basename = os.path.splitext(img_file)[0]
                # Find corresponding mask
                mask_file = None
                for mask_f in train_mask_files:
                    if os.path.splitext(mask_f)[0] == img_basename:
                        mask_file = mask_f
                        break
                
                if mask_file:
                    image_mask_pairs.append({
                        'image_file': img_file,
                        'mask_file': mask_file,
                        'basename': img_basename
                    })
            
            df = pd.DataFrame(image_mask_pairs)
            train_df, valid_df = self.split(df)
            
            # Create train directory
            train_img_dir = os.path.join(data_dir, "train", "images")
            train_mask_dir = os.path.join(data_dir, "train", "masks")
            os.makedirs(train_img_dir, exist_ok=True)
            os.makedirs(train_mask_dir, exist_ok=True)
            
            # Create validation directory  
            valid_img_dir = os.path.join(data_dir, "validation", "images")
            valid_mask_dir = os.path.join(data_dir, "validation", "masks")
            os.makedirs(valid_img_dir, exist_ok=True)
            os.makedirs(valid_mask_dir, exist_ok=True)
            
            # Copy training files
            for _, row in train_df.iterrows():
                shutil.copy(
                    os.path.join(train_images_dir, row['image_file']),
                    os.path.join(train_img_dir, row['image_file'])
                )
                shutil.copy(
                    os.path.join(train_masks_dir, row['mask_file']),
                    os.path.join(train_mask_dir, row['mask_file'])
                )
            
            # Copy validation files
            for _, row in valid_df.iterrows():
                shutil.copy(
                    os.path.join(train_images_dir, row['image_file']),
                    os.path.join(valid_img_dir, row['image_file'])
                )
                shutil.copy(
                    os.path.join(train_masks_dir, row['mask_file']),
                    os.path.join(valid_mask_dir, row['mask_file'])
                )
        
        # Create metadata.jsonl files for both splits
        def create_metadata(img_dir, mask_dir, split_name):
            img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(ALLOWED_EXTENSIONS)]
            metadata = []
            
            for img_file in img_files:
                img_basename = os.path.splitext(img_file)[0]
                # Find corresponding mask
                mask_file = None
                for f in os.listdir(mask_dir):
                    if f.lower().endswith('.png') and os.path.splitext(f)[0] == img_basename:
                        mask_file = f
                        break
                
                if mask_file:
                    # Include subdirectory paths for imagefolder loader
                    if img_dir != mask_dir:
                        # Different directories (images/ and masks/)
                        img_rel_path = f"images/{img_file}"
                        mask_rel_path = f"masks/{mask_file}"
                    else:
                        # Same directory (flat structure)
                        img_rel_path = img_file
                        mask_rel_path = mask_file
                    
                    metadata.append({
                        'file_name': img_file,
                        'image': img_rel_path,
                        'segmentation_mask': mask_rel_path
                    })
            
            # Save metadata
            metadata_df = pd.DataFrame(metadata)
            metadata_path = os.path.join(data_dir, split_name, "metadata.jsonl")
            metadata_df.to_json(metadata_path, orient='records', lines=True, force_ascii=False)
        
        # Create metadata for train split
        train_final_img_dir = os.path.join(data_dir, "train", "images")
        train_final_mask_dir = os.path.join(data_dir, "train", "masks")
        if not os.path.exists(train_final_img_dir):
            train_final_img_dir = os.path.join(data_dir, "train")
            train_final_mask_dir = os.path.join(data_dir, "train")
        
        create_metadata(train_final_img_dir, train_final_mask_dir, "train")
        
        # Create metadata for validation split if exists
        valid_final_img_dir = os.path.join(data_dir, "validation", "images")
        valid_final_mask_dir = os.path.join(data_dir, "validation", "masks")
        if os.path.exists(os.path.join(data_dir, "validation")):
            if not os.path.exists(valid_final_img_dir):
                valid_final_img_dir = os.path.join(data_dir, "validation")
                valid_final_mask_dir = os.path.join(data_dir, "validation")
            create_metadata(valid_final_img_dir, valid_final_mask_dir, "validation")
        
        # Create dataset directly instead of using imagefolder with metadata
        from datasets import Dataset, DatasetDict, Features, Image
        
        def load_segmentation_data(split_dir):
            
            metadata_path = os.path.join(split_dir, "metadata.jsonl")
            if not os.path.exists(metadata_path):
                return None
            
            # Read metadata
            metadata_df = pd.read_json(metadata_path, lines=True)
            
            # Create full paths
            data = []
            for _, row in metadata_df.iterrows():
                img_path = os.path.join(split_dir, row['image'])
                mask_path = os.path.join(split_dir, row['segmentation_mask'])
                
                if os.path.exists(img_path) and os.path.exists(mask_path):
                    data.append({
                        'autotrain_image': img_path,
                        'autotrain_label': mask_path
                    })
            
            # Create dataset with proper features for semantic segmentation
            # Images are loaded as regular images, masks are loaded as grayscale label maps
            features = Features({
                'autotrain_image': Image(),
                'autotrain_label': Image(mode='L')  # Grayscale for label maps
            })
            
            return Dataset.from_list(data, features=features)
        
        # Load train and validation splits
        train_dataset = load_segmentation_data(os.path.join(data_dir, "train"))
        valid_dataset = load_segmentation_data(os.path.join(data_dir, "validation"))
        
        if valid_dataset is not None:
            dataset = DatasetDict({
                "train": train_dataset,
                "validation": valid_dataset
            })
        else:
            dataset = DatasetDict({
                "train": train_dataset
            })
        
        # Save or push dataset
        if self.local:
            dataset.save_to_disk(f"{self.project_name}/autotrain-data")
            
            # Copy classes.txt to the output directory if it exists
            classes_file_source = self._find_classes_file(self.train_data)
            if classes_file_source:
                classes_file_dest = os.path.join(f"{self.project_name}/autotrain-data", "classes.txt")
                shutil.copy2(classes_file_source, classes_file_dest)
                print(f"Copied classes.txt to {classes_file_dest}")
        else:
            dataset.push_to_hub(
                f"{self.username}/autotrain-data-{self.project_name}",
                private=True,
                token=self.token,
            )
        
        if self.local:
            return f"{self.project_name}/autotrain-data"
        return f"{self.username}/autotrain-data-{self.project_name}"


@dataclass
class ImageInstanceSegmentationPreprocessor:
    """
    A class used to preprocess image data for instance segmentation tasks.
    
    Supports the standard instance segmentation format:
    - images/ folder with RGB images (JPG/PNG)
    - masks/ folder with instance masks (PNG files with unique instance IDs)
    - Optional annotations/ folder with bounding box and category information
    - Optional classes.txt or labelmap.txt with class names (one per line)
    
    Expected input structure (directory):
    ```
    dataset/
    ├── images/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    ├── masks/ 
    │   ├── image1.png
    │   ├── image2.png  
    │   └── ...
    ├── annotations/ (optional)
    │   ├── image1.json
    │   ├── image2.json
    │   └── ...
    └── classes.txt (optional)
    ```

    Attributes
    ----------
    train_data : str
        Path to the training data directory.
    username : str
        Username for the Hugging Face Hub.
    project_name : str
        Name of the project.
    token : str
        Authentication token for the Hugging Face Hub.
    valid_data : Optional[str], optional
        Path to the validation data directory, by default None.
    test_size : Optional[float], optional
        Proportion of the dataset to include in the validation split, by default 0.2.
    seed : Optional[int], optional
        Random seed for reproducibility, by default 42.
    local : Optional[bool], optional
        Whether to save the dataset locally or push to the Hugging Face Hub, by default False.

    Methods
    -------
    __post_init__():
        Validates the structure and contents of the training and validation data.
    prepare() -> str:
        Prepares the dataset for training and either saves it locally or pushes it to the Hugging Face Hub.
    """

    train_data: str
    username: str
    project_name: str
    token: str
    valid_data: Optional[str] = None
    test_size: Optional[float] = 0.2
    seed: Optional[int] = 42
    local: Optional[bool] = False

    @staticmethod
    def _find_images_and_masks_dirs(data_dir):
        """Find images and masks directories."""
        image_dir_names = ['images', 'img', 'image', 'imgs']
        mask_dir_names = ['masks', 'mask', 'instances', 'instance_masks']
        
        images_dir = None
        masks_dir = None
        
        for name in image_dir_names:
            potential_dir = os.path.join(data_dir, name)
            if os.path.exists(potential_dir):
                images_dir = potential_dir
                break
        
        for name in mask_dir_names:
            potential_dir = os.path.join(data_dir, name)
            if os.path.exists(potential_dir):
                masks_dir = potential_dir
                break
        
        if images_dir is None:
            raise ValueError(f"Could not find images directory in {data_dir}")
        
        if masks_dir is None:
            raise ValueError(f"Could not find masks directory in {data_dir}")
        
        return images_dir, masks_dir

    @staticmethod
    def _find_classes_file(data_dir):
        """Find and load classes file."""
        class_file_names = ['classes.txt', 'labelmap.txt', 'labels.txt']
        for class_file in class_file_names:
            class_path = os.path.join(data_dir, class_file)
            if os.path.exists(class_path):
                return class_path
        return None

    @staticmethod
    def _load_classes(data_dir):
        """Load class names from file."""
        class_file_path = ImageInstanceSegmentationPreprocessor._find_classes_file(data_dir)
        if class_file_path:
            with open(class_file_path, 'r') as f:
                classes = [line.strip() for line in f.readlines() if line.strip()]
            return classes
        return None

    @staticmethod
    def _get_unique_mask_values(mask_dir):
        """Get unique pixel values from all mask files to determine number of instances."""
        import numpy as np
        from PIL import Image as PILImage
        
        unique_values = set()
        mask_files = [f for f in os.listdir(mask_dir) if f.lower().endswith('.png')]
        
        for mask_file in mask_files[:10]:  # Sample first 10 masks
            mask_path = os.path.join(mask_dir, mask_file)
            mask = PILImage.open(mask_path)
            mask_array = np.array(mask)
            unique_values.update(np.unique(mask_array).tolist())
        
        # Remove background (0) and sort
        unique_values.discard(0)
        return sorted(list(unique_values))

    @staticmethod
    def _validate_dataset_structure(images_dir, masks_dir):
        """Validate that the dataset has proper structure."""
        if not os.path.exists(images_dir):
            raise ValueError(f"Images directory not found: {images_dir}")
        
        if not os.path.exists(masks_dir):
            raise ValueError(f"Masks directory not found: {masks_dir}")
        
        # Get image and mask files
        image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(ALLOWED_EXTENSIONS)]
        mask_files = [f for f in os.listdir(masks_dir) if f.lower().endswith('.png')]
        
        if len(image_files) == 0:
            raise ValueError(f"No image files found in {images_dir}")
        
        if len(mask_files) == 0:
            raise ValueError(f"No mask files found in {masks_dir}")
        
        # Check if we have corresponding masks for images
        image_basenames = {os.path.splitext(f)[0] for f in image_files}
        mask_basenames = {os.path.splitext(f)[0] for f in mask_files}
        
        missing_masks = image_basenames - mask_basenames
        if missing_masks:
            raise ValueError(f"Missing masks for images: {list(missing_masks)[:5]}...")
        
        return image_files, mask_files

    def __post_init__(self):
        if not os.path.exists(self.train_data):
            raise ValueError(f"{self.train_data} does not exist.")
        
        # Find and validate training data structure
        train_images_dir, train_masks_dir = self._find_images_and_masks_dirs(self.train_data)
        self._validate_dataset_structure(train_images_dir, train_masks_dir)
        
        # Validate validation data if provided
        if self.valid_data:
            if not os.path.exists(self.valid_data):
                raise ValueError(f"{self.valid_data} does not exist.")
            
            valid_images_dir, valid_masks_dir = self._find_images_and_masks_dirs(self.valid_data)
            self._validate_dataset_structure(valid_images_dir, valid_masks_dir)

    def split(self, df):
        """Split dataframe into train and validation sets."""
        train_df, valid_df = train_test_split(
            df,
            test_size=self.test_size,
            random_state=self.seed,
        )
        train_df = train_df.reset_index(drop=True)
        valid_df = valid_df.reset_index(drop=True)
        return train_df, valid_df

    def prepare(self):
        """
        Prepare the instance segmentation dataset following the standard format.
        Creates a Hugging Face dataset with images and instance masks.
        """
        from datasets import Dataset, DatasetDict, Features, Image
        import json
        
        random_uuid = uuid.uuid4()
        cache_dir = os.environ.get("HF_HOME")
        if not cache_dir:
            cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
        data_dir = os.path.join(cache_dir, "autotrain", str(random_uuid))
        
        # Process training data
        train_images_dir, train_masks_dir = self._find_images_and_masks_dirs(self.train_data)
        train_image_files, train_mask_files = self._validate_dataset_structure(train_images_dir, train_masks_dir)
        
        # Load class names if available
        classes = self._load_classes(self.train_data)
        if classes is None:
            # Create default classes
            classes = ["object"]  # Default single class for instance segmentation
        
        num_classes = len(classes)
        print(f"Found {num_classes} classes: {classes}")
        
        # Check for annotations directory
        annotations_dir = os.path.join(self.train_data, "annotations")
        has_annotations = os.path.exists(annotations_dir)
        
        if self.valid_data:
            # Use provided validation data
            shutil.copytree(self.train_data, os.path.join(data_dir, "train"))
            shutil.copytree(self.valid_data, os.path.join(data_dir, "validation"))
        else:
            # Create train/validation split
            image_mask_pairs = []
            for img_file in train_image_files:
                img_basename = os.path.splitext(img_file)[0]
                # Find corresponding mask
                mask_file = None
                for mask_f in train_mask_files:
                    if os.path.splitext(mask_f)[0] == img_basename:
                        mask_file = mask_f
                        break
                
                if mask_file:
                    entry = {
                        'image_file': img_file,
                        'mask_file': mask_file,
                        'basename': img_basename
                    }
                    
                    # Add annotation file if it exists
                    if has_annotations:
                        ann_file = f"{img_basename}.json"
                        ann_path = os.path.join(annotations_dir, ann_file)
                        if os.path.exists(ann_path):
                            entry['annotation_file'] = ann_file
                    
                    image_mask_pairs.append(entry)
            
            df = pd.DataFrame(image_mask_pairs)
            train_df, valid_df = self.split(df)
            
            # Create directories
            for split, split_df in [("train", train_df), ("validation", valid_df)]:
                split_dir = os.path.join(data_dir, split)
                split_img_dir = os.path.join(split_dir, "images")
                split_mask_dir = os.path.join(split_dir, "masks")
                os.makedirs(split_img_dir, exist_ok=True)
                os.makedirs(split_mask_dir, exist_ok=True)
                
                if has_annotations:
                    split_ann_dir = os.path.join(split_dir, "annotations")
                    os.makedirs(split_ann_dir, exist_ok=True)
                
                # Copy files
                for _, row in split_df.iterrows():
                    shutil.copy(
                        os.path.join(train_images_dir, row['image_file']),
                        os.path.join(split_img_dir, row['image_file'])
                    )
                    shutil.copy(
                        os.path.join(train_masks_dir, row['mask_file']),
                        os.path.join(split_mask_dir, row['mask_file'])
                    )
                    
                    if has_annotations and 'annotation_file' in row:
                        shutil.copy(
                            os.path.join(annotations_dir, row['annotation_file']),
                            os.path.join(split_ann_dir, row['annotation_file'])
                        )
        
        # Create dataset
        def load_instance_segmentation_data(split_dir):
            img_dir = os.path.join(split_dir, "images")
            mask_dir = os.path.join(split_dir, "masks")
            ann_dir = os.path.join(split_dir, "annotations")
            
            if not os.path.exists(img_dir):
                return None
            
            data = []
            img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(ALLOWED_EXTENSIONS)]
            
            for img_file in img_files:
                img_basename = os.path.splitext(img_file)[0]
                img_path = os.path.join(img_dir, img_file)
                mask_path = os.path.join(mask_dir, f"{img_basename}.png")
                
                if os.path.exists(mask_path):
                    entry = {
                        'autotrain_image': img_path,
                        'autotrain_instance_mask': mask_path,
                        'autotrain_bbox': [],  # Placeholder for bounding boxes
                        'autotrain_category': []  # Placeholder for categories
                    }
                    
                    # Load annotation if available
                    ann_path = os.path.join(ann_dir, f"{img_basename}.json")
                    if os.path.exists(ann_path):
                        try:
                            with open(ann_path, 'r') as f:
                                ann_data = json.load(f)
                            
                            # Extract bounding boxes and categories if available
                            if 'objects' in ann_data:
                                bboxes = []
                                categories = []
                                for obj in ann_data['objects']:
                                    if 'bbox' in obj:
                                        bboxes.append(obj['bbox'])
                                    if 'category' in obj:
                                        categories.append(obj['category'])
                                    elif 'class' in obj:
                                        categories.append(obj['class'])
                                
                                entry['autotrain_bbox'] = bboxes
                                entry['autotrain_category'] = categories
                        except Exception as e:
                            print(f"Warning: Could not parse annotation {ann_path}: {e}")
                    
                    data.append(entry)
            
            # Create dataset with proper features for instance segmentation
            from datasets import Sequence, Value
            features = Features({
                'autotrain_image': Image(),
                'autotrain_instance_mask': Image(mode='L'),  # Grayscale for instance masks
                'autotrain_bbox': Sequence(Sequence(Value('float32'), length=4)),  # List of bounding boxes [x, y, w, h]
                'autotrain_category': Sequence(Value('int32'))  # List of category IDs
            })
            
            return Dataset.from_list(data, features=features)
        
        # Load train and validation splits
        train_dataset = load_instance_segmentation_data(os.path.join(data_dir, "train"))
        valid_dataset = load_instance_segmentation_data(os.path.join(data_dir, "validation"))
        
        if valid_dataset is not None:
            dataset = DatasetDict({
                "train": train_dataset,
                "validation": valid_dataset
            })
        else:
            dataset = DatasetDict({
                "train": train_dataset
            })
        
        # Save or push dataset
        if self.local:
            dataset.save_to_disk(f"{self.project_name}/autotrain-data")
            
            # Copy classes.txt to the output directory if it exists
            classes_file_source = self._find_classes_file(self.train_data)
            if classes_file_source:
                classes_file_dest = os.path.join(f"{self.project_name}/autotrain-data", "classes.txt")
                shutil.copy2(classes_file_source, classes_file_dest)
                print(f"Copied classes.txt to {classes_file_dest}")
        else:
            dataset.push_to_hub(
                f"{self.username}/autotrain-data-{self.project_name}",
                private=True,
                token=self.token,
            )
        
        if self.local:
            return f"{self.project_name}/autotrain-data"
        return f"{self.username}/autotrain-data-{self.project_name}"
