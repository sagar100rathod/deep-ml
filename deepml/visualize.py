import os
from typing import Callable, List, Tuple

import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from deepml.utils import (
    get_random_samples_batch_from_dataset,
    get_random_samples_batch_from_loader,
    transform_input,
    transform_target,
)


def plot_images(
    images: List[np.ndarray],
    labels: List[str] = None,
    cols: int = 4,
    figsize: Tuple[int, int] = (10, 10),
    fontsize: int = 14,
):
    """Displays a grid of images with optional labels using matplotlib.

    Creates a multi-panel figure showing images in a grid layout with optional
    titles for each image.

    Args:
        images: List of images as numpy arrays in HWC or HW format.
        labels: List of labels/titles for each image. If provided, must have
            the same length as images. Defaults to None.
        cols: Number of columns in the grid. Rows are calculated automatically.
            Defaults to 4.
        figsize: Size of the matplotlib figure as (width, height) tuple.
            Defaults to (10, 10).
        fontsize: Font size for image titles. Defaults to 14.

    Note:
        The function automatically calculates the number of rows needed based
        on the number of images and columns. Axes ticks are hidden for cleaner
        visualization.
    """
    plt.figure(figsize=figsize)
    rows = int(np.ceil(len(images) / cols))
    for index, image in enumerate(images):
        ax = plt.subplot(rows, cols, index + 1, xticks=[], yticks=[])
        if labels:
            ax.set_title(labels[index])
        ax.title.set_fontsize(fontsize)
        plt.imshow(image)
    plt.tight_layout()


def plot_images_with_title(
    image_generator, samples: int, cols=4, figsize=(10, 10), fontsize=14
):
    """Displays a grid of images with colored titles using matplotlib.

    Creates a multi-panel figure showing images in a grid layout with titles
    that can have custom colors (useful for showing correct/incorrect predictions).

    Args:
        image_generator: Generator or iterable yielding tuples of
            (image, title, title_color) where:
                - image: numpy array in HWC or HW format
                - title: String title for the image
                - title_color: Optional color string (e.g., 'red', 'green', '#ff0000').
                  If None, uses default matplotlib text color.
        samples: Total number of images to display from the generator.
        cols: Number of columns in the grid. Defaults to 4.
        figsize: Size of the matplotlib figure as (width, height) tuple.
            Defaults to (10, 10).
        fontsize: Font size for image titles. Defaults to 14.

    Note:
        This function is commonly used for showing model predictions where
        title colors indicate correctness (green for correct, red for incorrect).
    """

    plt.figure(figsize=figsize)
    rows = int(np.ceil(samples / cols))
    for index, (image, title, title_color) in enumerate(image_generator):
        ax = plt.subplot(rows, cols, index + 1, xticks=[], yticks=[])
        ax.set_title(
            title,
            color=mpl.rcParams["text.color"] if title_color is None else title_color,
        )
        ax.title.set_fontsize(fontsize)
        ax.imshow(image)

    plt.show()
    plt.tight_layout()


def plot_images_with_bboxes(
    image_generator,
    samples: int,
    cols=4,
    figsize=(10, 10),
    fontsize=14,
    classes: List[str] = None,
    class_color_map: dict = None,
    cmap: str = "tab10",
):
    """Displays a grid of images with bounding boxes and class labels.

    Creates a multi-panel figure showing images with drawn bounding boxes and
    labeled class names. Each bounding box is colored based on its class.

    Args:
        image_generator: Generator or iterable yielding tuples of
            (image, title, bboxes) where:
                - image: numpy array in HWC or HW format
                - title: String title for the image
                - bboxes: List of bounding boxes, each as [class_id, xmin, ymin, width, height]
                  where class_id can be an integer index or string label.
        samples: Total number of images to display from the generator.
        cols: Number of columns in the grid. Defaults to 4.
        figsize: Size of the matplotlib figure as (width, height) tuple.
            Defaults to (10, 10).
        fontsize: Font size for image titles and bbox labels. Defaults to 14.
        classes: Optional list mapping class indices to class names. If provided
            and class_id is an integer, uses classes[class_id] as the label.
            Defaults to None.
        class_color_map: Optional dictionary mapping class IDs or names to color
            strings (e.g., '#ff0000', 'red'). If a class has no mapping, falls
            back to the colormap. Defaults to None.
        cmap: Matplotlib colormap name used as fallback for bbox colors when
            class_color_map doesn't provide a color. Defaults to "tab10".

    Note:
        Bounding boxes are drawn with red edges and labeled with a colored
        background box containing the class name. Label text is white for
        better visibility against the colored background.
    """

    plt.figure(figsize=figsize)
    rows = int(np.ceil(samples / cols))

    cmap_obj = mpl.cm.get_cmap(cmap)
    cmap_n = getattr(cmap_obj, "N", 10)

    def _get_color_for_class(key):
        # Try direct mapping by provided map (allow int or str keys)
        if class_color_map:
            if key in class_color_map:
                return class_color_map[key]
            key_str = str(key)
            if key_str in class_color_map:
                return class_color_map[key_str]

        # Fallback: derive an index from key (int or hashed) and use cmap
        try:
            idx = int(key)  # if key is numeric index
        except Exception:
            idx = abs(hash(str(key))) % cmap_n
        rgba = cmap_obj(idx % cmap_n)
        # convert rgba to hex for consistent use with bbox background
        return mpl.colors.to_hex(rgba)

    for index, (image, title, bboxes) in enumerate(image_generator):
        ax = plt.subplot(rows, cols, index + 1, xticks=[], yticks=[])
        ax.set_title(title, color=mpl.rcParams["text.color"])
        ax.title.set_fontsize(fontsize)
        ax.imshow(image)

        if bboxes is not None:
            for cls_id, xmin, ymin, width, height in bboxes:

                # resolve label text and color
                label_text = (
                    classes[cls_id]
                    if (classes and isinstance(cls_id, int) and cls_id < len(classes))
                    else str(cls_id)
                )
                color = _get_color_for_class(cls_id)

                # Create a rectangle patch
                rect = patches.Rectangle(
                    (xmin, ymin),
                    width,
                    height,
                    linewidth=1,
                    edgecolor="r",
                    facecolor="none",
                )

                ax.add_patch(rect)

                # Draw label box (ensure label sits inside image bounds)
                text_x = xmin
                text_y = max(ymin - 2, 0)
                ax.text(
                    text_x,
                    text_y,
                    label_text,
                    fontsize=max(int(fontsize * 0.7), 8),
                    color="white",
                    verticalalignment="bottom",
                    bbox=dict(facecolor=color, alpha=0.8, pad=0.2, edgecolor="none"),
                )
    plt.show()
    plt.tight_layout()


def show_images_from_loader(
    loader,
    image_inverse_transform=None,
    samples=9,
    cols=3,
    figsize=(5, 5),
    classes=None,
    title_color=None,
):
    """Displays random samples of images from a DataLoader.

    Randomly selects and displays images from a PyTorch DataLoader with their
    corresponding labels as titles.

    Args:
        loader: PyTorch DataLoader returning batches of (image, label) tensors.
        image_inverse_transform: Optional callable to reverse image normalization
            or transformations before display (e.g., denormalization). Defaults to None.
        samples: Number of images to display. Defaults to 9.
        cols: Number of columns in the grid. Defaults to 3.
        figsize: Size of the matplotlib figure as (width, height) tuple.
            Defaults to (5, 5).
        classes: Optional list of class names for converting label indices to
            text. If None and loader.dataset has a 'classes' attribute, uses that.
            Defaults to None.
        title_color: Optional color string for all image titles (e.g., 'blue').
            Defaults to None.

    Note:
        Images are randomly sampled from the DataLoader. If the DataLoader's
        dataset has a 'classes' attribute, it will be used automatically for
        label names unless overridden by the classes parameter.
    """
    x, y = get_random_samples_batch_from_loader(loader, samples=samples)
    x = transform_input(x, image_inverse_transform)

    if not classes and hasattr(loader.dataset, "classes"):
        classes = loader.dataset.classes

    image_title_generator = (
        (x[index], transform_target(y[index], classes), title_color)
        for index in range(x.shape[0])
    )
    plot_images_with_title(
        image_title_generator, samples=samples, cols=cols, figsize=figsize
    )


def show_images_from_dataset(
    dataset,
    image_inverse_transform=None,
    samples=9,
    cols=3,
    figsize=(10, 10),
    classes=None,
    title_color=None,
):
    """Displays random samples of images from a Dataset.

    Randomly selects and displays images from a PyTorch Dataset with their
    corresponding labels as titles.

    Args:
        dataset: PyTorch Dataset returning (image, label) tuples.
        image_inverse_transform: Optional callable to reverse image normalization
            or transformations before display (e.g., denormalization). Defaults to None.
        samples: Number of images to display. Defaults to 9.
        cols: Number of columns in the grid. Defaults to 3.
        figsize: Size of the matplotlib figure as (width, height) tuple.
            Defaults to (10, 10).
        classes: Optional list of class names for converting label indices to
            text. If None and dataset has a 'classes' attribute, uses that.
            Defaults to None.
        title_color: Optional color string for all image titles (e.g., 'blue').
            Defaults to None.

    Note:
        Images are randomly sampled from the Dataset. If the Dataset has a
        'classes' attribute, it will be used automatically for label names
        unless overridden by the classes parameter.
    """
    x, y = get_random_samples_batch_from_dataset(dataset, samples=samples)
    x = transform_input(x, image_inverse_transform)

    if not classes and hasattr(dataset, "classes"):
        classes = dataset.classes

    image_title_generator = (
        (x[index], transform_target(y[index], classes), title_color)
        for index in range(x.shape[0])
    )
    plot_images_with_title(
        image_title_generator, samples=samples, cols=cols, figsize=figsize
    )


def show_images_from_folder(
    img_dir,
    images=None,
    open_file_func: Callable = None,
    samples=9,
    cols=3,
    figsize=(10, 10),
    title_color=None,
):
    """Displays random samples of images from a folder.

    Randomly selects and displays images from a directory with filenames as titles.

    Args:
        img_dir: Directory path containing image files.
        images: Optional list of image filenames to display. If None, all files
            in img_dir are used and randomly sampled. Defaults to None.
        open_file_func: Optional callable to open image files. Should accept a
            file path and return an image object. If None, uses PIL.Image.open.
            Defaults to None.
        samples: Number of images to display. If fewer images exist, displays all.
            Defaults to 9.
        cols: Number of columns in the grid. Defaults to 3.
        figsize: Size of the matplotlib figure as (width, height) tuple.
            Defaults to (10, 10).
        title_color: Optional color string for all image titles (e.g., 'blue').
            Defaults to None.

    Note:
        If the number of requested samples exceeds available images, all images
        are displayed. Images are randomly sampled without replacement.
    """
    if not images:
        files = os.listdir(img_dir)
        if samples < len(files):
            images = np.random.choice(files, size=samples, replace=False)
        else:
            images = files

    open_file_func = Image.open if open_file_func is None else open_file_func
    image_generator = (
        (open_file_func(os.path.join(img_dir, file)), file, title_color)
        for file in images
    )
    plot_images_with_title(image_generator, len(images), cols=cols, figsize=figsize)


def show_images_from_dataframe(
    dataframe,
    img_dir=None,
    image_file_name_column="image",
    image_filepath_column=None,
    open_file_func: Callable = None,
    label_column: str = None,
    bbox_label_column: str = None,
    samples=9,
    cols=3,
    figsize=(10, 10),
    classes=None,
    class_color_map: dict = None,
    cmap: str = "tab10",
):
    """Displays random samples of images from a pandas DataFrame.

    Randomly selects and displays images specified in a DataFrame, with optional
    labels and bounding boxes.

    Args:
        dataframe: pandas DataFrame containing image file information.
        img_dir: Directory containing images. Required if image_filepath_column
            is not provided. Defaults to None.
        image_file_name_column: Column name containing image filenames (used with
            img_dir). Defaults to "image".
        image_filepath_column: Column name containing absolute image file paths.
            If provided, takes precedence over image_file_name_column and img_dir.
            Defaults to None.
        open_file_func: Optional callable to open image files. Should accept a
            file path and return an image object. If None, uses PIL.Image.open.
            Defaults to None.
        label_column: Column name containing image labels. If None, displays row
            indices instead. Defaults to None.
        bbox_label_column: Column name containing bounding box data. Each entry
            should be a list of bounding boxes in format [class_id, xmin, ymin,
            width, height]. If provided, displays images with bounding boxes.
            Defaults to None.
        samples: Number of random images to display from the DataFrame.
            Defaults to 9.
        cols: Number of columns in the grid. Defaults to 3.
        figsize: Size of the matplotlib figure as (width, height) tuple.
            Defaults to (10, 10).
        classes: Optional list mapping class indices to class names for bbox
            labels. Defaults to None.
        class_color_map: Optional dictionary mapping class IDs or names to color
            strings (e.g., '#ff0000', 'red'). Used for bbox colors. Defaults to None.
        cmap: Matplotlib colormap name used as fallback for bbox colors when
            class_color_map doesn't provide a color. Defaults to "tab10".

    Note:
        - If bbox_label_column is provided, displays images with bounding boxes
          using plot_images_with_bboxes.
        - Otherwise, displays images with titles using plot_images_with_title.
        - Images are randomly sampled from the DataFrame.
    """
    samples = dataframe.sample(samples)
    open_file_func = Image.open if open_file_func is None else open_file_func
    image_generator = (
        (
            (
                open_file_func(row_data[image_filepath_column])
                if image_filepath_column
                else open_file_func(
                    os.path.join(img_dir, row_data[image_file_name_column])
                )
            ),
            f"idx: {row_idx}" if label_column is None else row_data["label_column"],
            None if bbox_label_column is None else row_data[bbox_label_column],
        )
        for row_idx, row_data in samples.iterrows()
    )

    if bbox_label_column is not None:
        plot_images_with_bboxes(
            image_generator,
            len(samples),
            cols=cols,
            figsize=figsize,
            classes=classes,
            class_color_map=class_color_map,
            cmap=cmap,
        )
    else:
        plot_images_with_title(
            image_generator, len(samples), cols=cols, figsize=figsize
        )
