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
    """
    Display a grid of images with optional labels using matplotlib.

    Args:
        images (List[np.ndarray]): List of images as numpy arrays.
        labels (List[str], optional): List of labels for each image. Defaults to None.
        cols (int, optional): Number of columns in the grid. Defaults to 4.
        figsize (Tuple[int, int], optional): Size of the matplotlib figure. Defaults to (10, 10).
        fontsize (int, optional): Font size for image titles. Defaults to 14.

    Returns:
        None
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
    """
    Display a grid of images with colored titles using matplotlib.

    Args:
        image_generator: Generator yielding tuples (image: np.ndarray, title: str, title_color: Optional[str]).
                        The title color is optional.
        samples (int): Total number of images to display.
        cols (int, optional): Number of columns in the grid. Defaults to 4.
        figsize (Tuple[int, int], optional): Size of the matplotlib figure. Defaults to (10, 10).
        fontsize (int, optional): Font size for image titles. Defaults to 14.

    Returns:
        None
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
    """
    Display a grid of images with bounding boxes and class labels using matplotlib.

    Args:
        image_generator: Generator yielding tuples (image: np.ndarray, title: str, bboxes: List[List[float]]).
                         Each bbox is [id, xmin, ymin, width, height] where id may be an int index or label.
        samples (int): Total number of images to display.
        cols (int, optional): Number of columns in the grid. Defaults to 4.
        figsize (Tuple[int, int], optional): Size of the matplotlib figure. Defaults to (10, 10).
        fontsize (int, optional): Font size for image titles and bbox labels. Defaults to 14.
        classes (List[str], optional): Optional list mapping class indices to class names.
        class_color_map (dict, optional): Mapping from class id or class name to a color string (e.g. '#ff0000').
        cmap (str, optional): Matplotlib colormap name used as fallback when class_color_map does not provide a color.

    Returns:
        None
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
    """
    Display random samples of images from a DataLoader using matplotlib.

    Args:
        loader: torch.utils.data.DataLoader returning image and label tensors.
        image_inverse_transform (callable, optional): Function to inverse-transform images before display.
        samples (int, optional): Number of images to display. Defaults to 9.
        cols (int, optional): Number of columns in the grid. Defaults to 3.
        figsize (Tuple[int, int], optional): Size of the matplotlib figure. Defaults to (5, 5).
        classes (List[str], optional): List of class names for labels.
        title_color (str, optional): Color for image titles.

    Returns:
        None
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
    """
    Display random samples of images from a Dataset using matplotlib.

    Args:
        dataset: torch.utils.data.Dataset returning image and label tensors.
        image_inverse_transform (callable, optional): Function to inverse-transform images before display.
        samples (int, optional): Number of images to display. Defaults to 9.
        cols (int, optional): Number of columns in the grid. Defaults to 3.
        figsize (Tuple[int, int], optional): Size of the matplotlib figure. Defaults to (10, 10).
        classes (List[str], optional): List of class names for labels.
        title_color (str, optional): Color for image titles.

    Returns:
        None
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
    """
    Display random samples of images from a folder or list using matplotlib.

    Args:
        img_dir (str): Directory containing image files.
        images (List[str], optional): List of image filenames. If None, all files in img_dir are used.
        open_file_func (callable, optional): Function to open image files. Defaults to PIL.Image.open.
        samples (int, optional): Number of images to display. Defaults to 9.
        cols (int, optional): Number of columns in the grid. Defaults to 3.
        figsize (Tuple[int, int], optional): Size of the matplotlib figure. Defaults to (10, 10).
        title_color (str, optional): Color for image titles.

    Returns:
        None
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
    """
    Display random samples of images from a DataFrame using matplotlib.

    Args:
        dataframe (pd.DataFrame): DataFrame containing image file info.
        img_dir (str, optional): Directory containing images. If None, file paths must be absolute.
        image_file_name_column (str, optional): Column name for image filenames. Defaults to "image".
        image_filepath_column (str, optional): Column name for absolute image file paths.
        open_file_func (callable, optional): Function to open image files. Defaults to PIL.Image.open.
        label_column (str, optional): Column name for image labels.
        bbox_label_column (str, optional): Column name for bounding boxes (list of [id, x_min, y_min, width, height]).
        samples (int, optional): Number of images to display. Defaults to 9.
        cols (int, optional): Number of columns in the grid. Defaults to 3.
        figsize (Tuple[int, int], optional): Size of the matplotlib figure. Defaults to (10, 10).
        classes (List[str], optional): List of class names for bbox labels.
        class_color_map (dict, optional): Mapping from class id or class name to a color string (e.g. '#ff0000').
        cmap (str, optional): Matplotlib colormap name used as fallback when class_color_map does not provide a color.

    Returns:
        None
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
