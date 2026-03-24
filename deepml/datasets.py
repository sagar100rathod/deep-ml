import os
from typing import Any, Callable, List, Tuple, Union

import numpy as np
import pandas as pd
import torch
from PIL import Image


class ImageRowDataFrameDataset(torch.utils.data.Dataset):
    """Dataset for reading images stored as flattened arrays in DataFrame rows.

    This dataset treats each row of a DataFrame as a flattened image array,
    which is then reshaped to the specified image dimensions.

    Attributes:
        dataframe: DataFrame containing flattened image data (without target column).
        target_column: Series containing target labels, if provided.
        samples: Number of samples in the dataset.
        image_size: Tuple specifying the output image dimensions (height, width).
        transform: Optional transformation callable to apply to images.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        target_column: str = None,
        image_size: Tuple[int, int] = (28, 28),
        transform: Callable = None,
    ):
        """Initializes the ImageRowDataFrameDataset.

        Args:
            dataframe: DataFrame where each row contains a flattened image array.
            target_column: Name of the column containing target labels. If provided,
                this column is extracted and removed from the DataFrame. Defaults to None.
            image_size: Dimensions to reshape each image to as (height, width).
                Defaults to (28, 28).
            transform: Optional callable to transform images (e.g., torchvision transforms).
                Defaults to None.

        Note:
            The DataFrame is reset with a fresh index, and the target column (if specified)
            is removed from the image data.
        """
        self.dataframe = dataframe.reset_index(drop=True, inplace=False)
        self.target_column = None

        if target_column:
            self.target_column = self.dataframe[target_column]
            self.dataframe.drop(target_column, axis=1, inplace=True)

        self.samples = self.dataframe.shape[0]
        self.image_size = image_size
        self.transform = transform

    def __getitem__(self, index: int):
        """Retrieves an image and its label at the specified index.

        Args:
            index: Index of the sample to retrieve.

        Returns:
            Tuple of (image, label) where:
                - image: Transformed PIL Image or tensor of shape specified by image_size
                - label: Target label if target_column was provided, otherwise 0
        """

        X = self.dataframe.iloc[index]

        X = Image.fromarray(X.to_numpy().reshape(self.image_size).astype(np.uint8))
        if self.transform is not None:
            X = self.transform(X)

        y = 0
        if self.target_column is not None:
            y = self.target_column.loc[index]

        return X, y

    def __len__(self):
        """Returns the total number of samples in the dataset.

        Returns:
            Number of samples.
        """
        return self.samples


class ImageDataFrameDataset(torch.utils.data.Dataset):
    """Dataset for reading images from file paths specified in a DataFrame.

    This dataset loads images from disk based on file paths listed in a DataFrame,
    making it suitable for image classification and regression tasks.

    Attributes:
        dataframe: DataFrame containing image file paths and optional target columns.
        image_file_name_column: Name of the column containing image filenames.
        target_columns: Column name(s) containing target values.
        image_dir: Base directory containing images.
        transforms: Transformation callable to apply to images.
        samples: Number of samples in the dataset.
        target_transform: Transformation callable to apply to targets.
        open_file_func: Custom function for opening image files.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        image_file_name_column: str = "image",
        target_columns: Union[int, List[str]] = None,
        image_dir: str = None,
        transforms: Callable = None,
        target_transform: Callable = None,
        open_file_func: Callable = None,
    ):
        """Initializes the ImageDataFrameDataset.

        Args:
            dataframe: DataFrame containing image file paths and optional targets.
            image_file_name_column: Name of the column containing image filenames.
                Defaults to "image".
            target_columns: Column name(s) containing target values. Can be a single
                column name (str) or list of column names for multi-target tasks.
                If None, no targets are loaded. Defaults to None.
            image_dir: Base directory containing images. If provided, filenames from
                the DataFrame are joined with this directory. Defaults to None.
            transforms: Optional callable to transform images (e.g., torchvision transforms).
                Defaults to None.
            target_transform: Optional callable to transform target values.
                Defaults to None.
            open_file_func: Custom callable to open image files. Should accept a file path
                and return an image object. If None, uses PIL.Image.open. Defaults to None.

        Note:
            The DataFrame is reset with a fresh index to ensure consistent indexing.
        """

        self.dataframe = dataframe.reset_index(drop=True, inplace=False)
        self.image_file_name_column = image_file_name_column
        self.target_columns = target_columns
        self.image_dir = image_dir
        self.transforms = transforms
        self.samples = self.dataframe.shape[0]
        self.target_transform = target_transform
        self.open_file_func = open_file_func

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset.

        Returns:
            Number of samples.
        """
        return self.samples

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Any]:
        """Retrieves an image and its target at the specified index.

        Args:
            index: Index of the sample to retrieve.

        Returns:
            Tuple of (image, target) where:
                - image: Transformed image as PIL Image or tensor
                - target: Target value(s) as tensor if target_columns was provided,
                  otherwise 0

        Note:
            If image_dir is provided, the image path is constructed by joining
            image_dir with the filename from the DataFrame.
        """

        image_file = self.dataframe.loc[index, self.image_file_name_column]

        if self.image_dir:
            image_file = os.path.join(self.image_dir, image_file)

        if self.open_file_func is None:
            X = Image.open(image_file)
        else:
            X = self.open_file_func(image_file)

        if self.transforms is not None:
            X = self.transforms(X)

        y = 0
        if self.target_columns:
            y = torch.tensor(self.dataframe.loc[index, self.target_columns])
            if self.target_transform:
                y = self.target_transform(y)

        return X, y


class ImageListDataset(torch.utils.data.Dataset):
    """Dataset for loading all images from a directory.

    This dataset reads all files from a specified directory and treats them as images.
    It returns both the image and its filename, making it useful for inference or
    unlabeled image processing tasks.

    Attributes:
        image_dir: Directory path containing image files.
        images: List of image filenames in the directory.
        transforms: Optional transformation callable to apply to images.
        open_file_func: Custom function for opening image files.
    """

    def __init__(
        self,
        image_dir: str,
        transforms: Callable = None,
        open_file_func: Callable = None,
    ):
        """Initializes the ImageListDataset.

        Args:
            image_dir: Directory path containing image files.
            transforms: Optional callable to transform images (e.g., torchvision transforms).
                Defaults to None.
            open_file_func: Custom callable to open image files. Should accept a file path
                and return an image object. If None, uses PIL.Image.open. Defaults to None.

        Note:
            All files in the directory are assumed to be images. No filtering is applied.
        """

        self.image_dir = image_dir
        self.images = os.listdir(image_dir)
        self.transforms = transforms
        self.open_file_func = open_file_func

    def __len__(self):
        """Returns the total number of images in the directory.

        Returns:
            Number of images.
        """
        return len(self.images)

    def __getitem__(self, index: int):
        """Retrieves an image and its filename at the specified index.

        Args:
            index: Index of the image to retrieve.

        Returns:
            Tuple of (image, filename) where:
                - image: Transformed image as PIL Image or tensor
                - filename: String filename of the image
        """

        image_file = self.images[index]
        if self.open_file_func is None:
            X = Image.open(os.path.join(self.image_dir, image_file))
        else:
            X = self.open_file_func(os.path.join(self.image_dir, image_file))

        if self.transforms is not None:
            X = self.transforms(X)

        return X, image_file


class SegmentationDataFrameDataset(torch.utils.data.Dataset):
    """Dataset for semantic segmentation with images and corresponding masks.

    This dataset loads images and their corresponding segmentation masks from
    directories specified in a DataFrame. It supports both training mode (with masks)
    and inference mode (without masks).

    Attributes:
        dataframe: DataFrame containing image and mask file information.
        image_dir: Directory containing input images.
        mask_dir: Directory containing segmentation masks (required for training).
        image_col: Column name for image filenames.
        mask_col: Column name for mask filenames.
        albu_torch_transforms: Albumentations transforms for augmentation.
        target_transform: Additional transforms for masks only.
        samples: Number of samples in the dataset.
        train: Whether the dataset is in training mode.
        open_file_func: Custom function for opening image files.

    Note:
        Image and mask files should have the same name unless mask_col specifies
        a different column. The open_file_func should accept an image_file_path
        and return a numpy array or PIL Image.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        image_dir: str,
        mask_dir: str = None,
        image_col: str = "image",
        mask_col: str = None,
        albu_torch_transforms: Callable = None,
        target_transform: Callable = None,
        train: bool = True,
        open_file_func: Callable = None,
    ):
        """Initializes the SegmentationDataFrameDataset.

        Args:
            dataframe: DataFrame containing image and mask file information.
            image_dir: Directory path containing input images.
            mask_dir: Directory path containing segmentation masks. Required when
                train=True. Defaults to None.
            image_col: Name of the DataFrame column containing image filenames.
                Defaults to "image".
            mask_col: Name of the DataFrame column containing mask filenames. If None,
                uses the same filenames as image_col. Defaults to None.
            albu_torch_transforms: Albumentations transforms to apply to both image
                and mask. Should return a dictionary with "image" and "mask" keys.
                Defaults to None.
            target_transform: Additional transform to apply only to the mask after
                albumentations transforms. Defaults to None.
            train: Whether the dataset is in training mode. If True, loads and returns
                masks. If False, returns filenames instead of masks. Defaults to True.
            open_file_func: Custom callable to open image/mask files. Should accept a
                file path and return a numpy array. If None, uses PIL.Image.open with
                conversion to numpy array. Defaults to None.

        Raises:
            AssertionError: If train=True and mask_dir is None.

        Note:
            - The DataFrame is reset with a fresh index for consistent indexing
            - In training mode, returns (image, mask) tuples
            - In inference mode, returns (image, filename) tuples
        """
        if train:
            assert mask_dir, "For training purpose, mask_dir should not be None"

        self.dataframe = dataframe.reset_index(drop=True, inplace=False)
        self.image_dir = image_dir
        self.mask_dir = mask_dir

        self.image_col = image_col
        self.mask_col = mask_col if mask_col else image_col

        self.albu_torch_transforms = albu_torch_transforms
        self.target_transform = target_transform
        self.samples = self.dataframe.shape[0]
        self.train = train
        self.open_file_func = open_file_func

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset.

        Returns:
            Number of samples.
        """
        return self.samples

    def __getitem__(self, index: int):
        """Retrieves an image and its mask (or filename) at the specified index.

        Args:
            index: Index of the sample to retrieve.

        Returns:
            Tuple of (image, target) where:
                - image: Transformed image tensor from albumentations
                - target: If train=True, transformed mask tensor. If train=False,
                  string filename of the image.

        Note:
            - In training mode, applies albumentations transforms to both image and mask
            - In inference mode, applies albumentations transforms only to image
            - Additional target_transform is applied to mask if provided (training only)
        """

        image_file = os.path.join(
            self.image_dir, self.dataframe.loc[index, self.image_col]
        )
        mask_file = (
            os.path.join(self.mask_dir, self.dataframe.loc[index, self.mask_col])
            if self.train
            else None
        )

        if self.open_file_func is None:
            image = np.array(Image.open(image_file))
            mask = np.array(Image.open(mask_file)) if self.train else None
        else:
            image = self.open_file_func(image_file)
            mask = self.open_file_func(mask_file) if self.train else None

        if self.train:
            transformed = self.albu_torch_transforms(image=image, mask=mask)
        else:
            transformed = self.albu_torch_transforms(image=image)

        if self.train and self.target_transform:
            transformed["mask"] = self.target_transform(transformed["mask"])

        if self.train:
            return transformed["image"], transformed["mask"]
        else:
            return transformed["image"], self.dataframe.loc[index, self.image_col]
