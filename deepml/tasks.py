import os
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from tqdm import tqdm

from deepml.tracking import MLExperimentLogger
from deepml.utils import blend, create_text_image, get_random_samples_batch_from_loader
from deepml.visualize import plot_images, plot_images_with_title


class Task(ABC):
    """Abstract base class for all deep learning tasks.

    This class provides the foundation for task-specific implementations including
    model management, device handling, and prediction workflows.

    Subclasses must implement methods for transforming targets and outputs, batch prediction, training and
    evaluation steps, and visualization.

    """

    def __init__(
        self,
        model: torch.nn.Module,
        model_dir: str,
        load_saved_model: bool = False,
        model_file_name: str = "latest_model.pt",
        device: str = "auto",
    ):
        """Initializes the Task.

        Args:
            model: PyTorch model instance to be trained or used for inference.
            model_dir: Directory path for saving and loading model checkpoints.
            load_saved_model: Whether to load a previously saved model from
                model_dir. Defaults to False.
                Set to True if you want to load model weights from a checkpoint file in model_dir.

            model_file_name: Name of the model checkpoint file.
                Defaults to "latest_model.pt".

            device: Device to use for computation. Options: "auto", "cpu",
                "cuda", or "mps". When "auto", automatically selects the best
                available device. Defaults to "auto".

        Raises:
            AssertionError: If model is not a torch.nn.Module instance, or if
                model_dir is None, or if model_file_name is not a string, or
                if device is not one of the valid options.
        """

        super(Task, self).__init__()

        assert isinstance(
            model, torch.nn.Module
        ), "model should be an instance of torch.nn.Module"
        assert model_dir is not None, "model_dir should not be None"
        assert isinstance(model_file_name, str), "model_file_name should be a string"
        assert device in [
            "auto",
            "cpu",
            "cuda",
            "mps",
        ], "device should be one of 'auto', 'cpu', 'cuda', 'mps'"

        self._model = model
        self._model_dir = model_dir
        self._model_file_name = model_file_name

        if device == "auto":
            if torch.cuda.is_available():
                self._device = torch.device("cuda")
            elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
                self._device = torch.device("mps")
            else:
                self._device = torch.device("cpu")
        else:
            self._device = torch.device(device)

        os.makedirs(self.model_dir, exist_ok=True)
        weights_file_path = os.path.join(self._model_dir, self._model_file_name)
        if load_saved_model:
            self.__load_model_weights(weights_file_path)

    def __load_model_weights(self, weights_file_path: str):
        if weights_file_path and os.path.exists(weights_file_path):
            print(f"Loading Saved Model Weights: {weights_file_path}")
            state_dict = torch.load(weights_file_path, map_location=self._device)
            self._model.load_state_dict(state_dict["model_state_dict"])
            print("Model Weights Successfully Loaded!")
        else:
            print("Failed to load model weights..!")

    @property
    def model(self):
        return self._model

    @property
    def model_dir(self):
        return self._model_dir

    @property
    def device(self):
        return self._device

    @property
    def model_file_name(self):
        return self._model_file_name

    def move_input_to_device(
        self,
        x: Union[torch.Tensor, list, tuple, dict],
        device: Union[torch.device, str, None] = None,
        non_blocking: bool = False,
        **kwargs: dict,
    ) -> Union[torch.Tensor, list, tuple, dict]:
        """Moves input data to the specified device.

        Handles various input types including tensors, lists, tuples, and
        dictionaries containing tensors.

        Args:
            x: Input data to move. Can be a single tensor, list/tuple of tensors,
                or dictionary with tensor values.
            device: Target device. If None, uses the task's default device.
                Defaults to None.
            non_blocking: Whether to use asynchronous transfer. Defaults to False.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            Input data moved to the target device, maintaining the original
            data structure.
        """

        if device is None:
            device = self._device

        if isinstance(x, torch.Tensor):
            x = x.to(device, non_blocking=non_blocking)
        elif isinstance(x, list):  # list of torch tensors
            x = [
                (
                    i.to(device, non_blocking=non_blocking)
                    if isinstance(i, torch.Tensor)
                    else i
                )
                for i in x
            ]
        elif isinstance(x, tuple):  # tuple of torch tensors
            x = tuple(
                [
                    (
                        i.to(device, non_blocking=non_blocking)
                        if isinstance(i, torch.Tensor)
                        else i
                    )
                    for i in x
                ]
            )
        elif isinstance(x, dict):  # dict values as torch tensors
            x = {
                key: (
                    value.to(device, non_blocking=non_blocking)
                    if isinstance(value, torch.Tensor)
                    else value
                )
                for key, value in x.items()
            }

        return x

    def transform_input(
        self, x: torch.Tensor, image_inverse_transform: Callable = None
    ) -> torch.Tensor:
        """Applies optional inverse transformation to input images.

        Args:
            x: Input image batch in BCHW format.
            image_inverse_transform: Optional transformation function to apply
                (e.g., denormalization). Defaults to None.

        Returns:
            Transformed image batch in BCHW format.
        """
        if image_inverse_transform is not None:
            x = image_inverse_transform(x)
        return x

    @abstractmethod
    def transform_target(self, y):
        """Transforms target data for visualization or evaluation.

        Args:
            y: Target data in model format.

        Returns:
            Transformed target data.
        """
        pass

    @abstractmethod
    def transform_output(self, prediction):
        """Transforms model output for visualization or evaluation.

        Args:
            prediction: Model output in raw format.

        Returns:
            Transformed prediction data.
        """
        pass

    @abstractmethod
    def predict_batch(self, x, *args, **kwargs):
        """Performs prediction on a single batch.

        Args:
            x: Input batch.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Model predictions for the batch.
        """
        pass

    @abstractmethod
    def train_step(self, x, y, *args, **kwargs) -> Tuple[Any, Any, Any]:
        """Executes a single training step.
           Apply any batch based transformation to the target as well, if needed.
        Args:
            x: Input batch.
            y: Target batch.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Tuple of (predictions, processed_inputs, processed_targets).
        """
        pass

    @abstractmethod
    def eval_step(self, x, y, *args, **kwargs) -> Tuple[Any, Any, Any]:
        """Executes a single evaluation step.

        Args:
            x: Input batch.
            y: Target batch.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Tuple of (predictions, processed_inputs, processed_targets).
        """
        pass

    @abstractmethod
    def predict(self, loader):
        """Generates predictions for all data in the loader.

        Args:
            loader: DataLoader containing data for prediction.

        Returns:
            Predictions and targets.
        """
        pass

    @abstractmethod
    def predict_class(self, loader):
        """Generates class predictions for all data in the loader.

        Args:
            loader: DataLoader containing data for prediction.

        Returns:
            Predicted classes, probabilities, and targets.
        """
        pass

    @abstractmethod
    def show_predictions(
        self,
        loader,
        image_inverse_transform=None,
        samples=9,
        cols=3,
        figsize=(10, 10),
        target_known=True,
    ):
        """Visualizes model predictions.

        Args:
            loader: DataLoader containing data for visualization.
            image_inverse_transform: Transformation to reverse normalization.
            samples: Number of samples to display.
            cols: Number of columns in visualization grid.
            figsize: Figure size tuple.
            target_known: Whether ground truth is available.
        """
        pass

    @abstractmethod
    def write_prediction_to_logger(
        self, tag, loader, logger, image_inverse_transform, global_step, img_size=224
    ):
        """Writes predictions to experiment logger.

        Args:
            tag: Tag identifier for logged data.
            loader: DataLoader containing data.
            logger: Experiment logger instance.
            image_inverse_transform: Transformation to reverse normalization.
            global_step: Current training step/epoch.
            img_size: Image size for logging.
        """
        pass

    @abstractmethod
    def evaluate(
        self,
        loader: torch.utils.data.DataLoader,
        criterion: torch.nn.Module,
        metrics: Dict[str, torch.nn.Module] = None,
        non_blocking=False,
    ):
        """Evaluates model performance on the given data.

        Args:
            loader: DataLoader containing evaluation data.
            criterion: Loss function module.
            metrics: Dictionary of metric modules.
            non_blocking: Whether to use async CUDA transfers.

        Returns:
            Dictionary of evaluation metrics.
        """
        pass


class NeuralNetTask(Task):
    """Base task implementation for general deep learning tasks.

    This class provides a simple implementation suitable for any deep learning task.
    It performs predictions without applying task-specific transformations and does
    not write to TensorBoard by default.

    Use this class when you need a minimal task implementation without specialized
    handling for classification, segmentation, or regression.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        model_dir: str,
        load_saved_model: bool = False,
        model_file_name: str = "latest_model.pt",
        device: str = "auto",
    ):
        """Initializes the NeuralNetTask.

        Args:
            model: PyTorch model instance to be trained or used for inference.
            model_dir: Directory path for saving and loading model checkpoints.
            load_saved_model: Whether to load a previously saved model from
                model_dir. Defaults to False.
            model_file_name: Name of the model checkpoint file.
                Defaults to "latest_model.pt".
            device: Device to use for computation. Options: "auto", "cpu",
                "cuda", or "mps". Defaults to "auto".
        """
        super(NeuralNetTask, self).__init__(
            model, model_dir, load_saved_model, model_file_name, device
        )

    def predict_batch(self, x: torch.Tensor, *args, **kwargs):
        """Performs prediction on a single batch.

        Args:
            x: Input batch tensor.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments. If 'model' key is present,
                uses that model instead of the task's default model.

        Returns:
            Model predictions for the batch.
        """
        x = self.move_input_to_device(x, **kwargs)

        if "model" in kwargs:
            return kwargs["model"](x)
        else:
            return self._model(x)

    def train_step(self, x, y, *args, **kwargs):
        """Executes a single training step.

        Args:
            x: Input batch.
            y: Target batch.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Tuple of (predictions, inputs, targets).
        """
        return self.predict_batch(x, *args, **kwargs), x, y

    def eval_step(self, x, y, *args, **kwargs):
        """Executes a single evaluation step.

        Args:
            x: Input batch.
            y: Target batch.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Tuple of (predictions, inputs, targets).
        """
        return self.predict_batch(x, *args, **kwargs), x, y

    def predict(self, loader: torch.utils.data.DataLoader):
        """Generates predictions for all batches in the data loader.

        Args:
            loader: DataLoader containing data for prediction.

        Returns:
            Tuple of (predictions, targets) where:
                - predictions: Concatenated tensor of all model predictions
                - targets: Concatenated tensor or list of all ground truth labels

        Raises:
            AssertionError: If loader is None or empty.
        """

        assert loader is not None and len(loader) > 0
        self._model.eval()
        self._model = self._model.to(self._device)
        predictions = []
        targets = []
        with torch.no_grad():
            for x, y in tqdm(
                loader, total=len(loader), desc="{:12s}".format("Prediction")
            ):
                y_pred, x, y = self.eval_step(x, y)
                predictions.append(y_pred)
                targets.append(y)

        predictions = torch.cat(predictions)
        targets = (
            torch.cat(targets)
            if isinstance(targets[0], torch.Tensor)
            else np.hstack(targets).tolist()
        )

        return predictions, targets

    def predict_class(self, loader: torch.utils.data.DataLoader):
        """Generates class predictions for all data in the loader.

        Args:
            loader: DataLoader containing data for prediction.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError()

    def show_predictions(
        self,
        loader: torch.utils.data.DataLoader,
        image_inverse_transform: Callable = None,
        samples: int = 9,
        cols: int = 3,
        figsize: Tuple[int, int] = (10, 10),
        target_known: bool = True,
    ):
        """Visualizes model predictions.

        Args:
            loader: DataLoader containing data for visualization.
            image_inverse_transform: Transformation to reverse normalization.
            samples: Number of samples to display.
            cols: Number of columns in visualization grid.
            figsize: Figure size tuple.
            target_known: Whether ground truth is available.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError()

    def transform_target(self, y: Any):
        """Transforms target data for visualization or evaluation.

        Args:
            y: Target data in model format.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError()

    def transform_output(self, prediction):
        """Transforms model output for visualization or evaluation.

        Args:
            prediction: Model output in raw format.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError()

    def write_prediction_to_logger(
        self,
        tag: str,
        loader,
        logger,
        image_inverse_transform,
        global_step,
        img_size=224,
        **kwargs: dict,
    ):
        """Writes predictions to experiment logger.

        Args:
            tag: Tag identifier for logged data.
            loader: DataLoader containing data.
            logger: Experiment logger instance.
            image_inverse_transform: Transformation to reverse normalization.
            global_step: Current training step/epoch.
            img_size: Image size for logging.
            **kwargs: Additional keyword arguments.

        Note:
            Default implementation does nothing. Override in subclasses for
            custom logging behavior.
        """
        pass

    @torch.no_grad()
    def evaluate(
        self,
        loader: torch.utils.data.DataLoader,
        metrics: Dict[str, torch.nn.Module] = None,
        non_blocking=False,
    ):
        """Evaluates the model on the given data loader using specified metrics.

        Args:
            loader: DataLoader containing evaluation data.
            metrics: Dictionary mapping metric names to metric modules. Each
                metric should be a torch.nn.Module with a forward() method.
                Defaults to None.
            non_blocking: Whether to use asynchronous CUDA transfers.
                Defaults to False.

        Returns:
            Dictionary mapping metric names to their average values across
            all batches.

        Raises:
            Exception: If loader is None.
        """
        if loader is None:
            raise Exception("Loader cannot be None.")

        self._model.eval()
        metrics_dict = {metric_name: 0.0 for metric_name in metrics.keys()}

        bar = tqdm(
            total=len(loader), desc="{:12s}".format("Evaluation"), dynamic_ncols=True
        )

        total_samples = 0
        for batch_index, (x, y) in enumerate(loader):

            outputs, x, y = self.eval_step(x, y, non_blocking)

            if isinstance(y, torch.Tensor):
                y = y.to(self._device)

            if (
                isinstance(outputs, torch.Tensor)
                and outputs.ndim == 2
                and outputs.shape[1] == 1
            ):
                y = y.view_as(outputs)

            batch_size = x.size(0)
            total_samples += batch_size

            for metric_name, metric_instance in metrics.items():
                metric_value = metric_instance(outputs, y).item()
                metrics_dict[metric_name] += metric_value * batch_size

            bar.update(1)

        bar.close()

        for metric_name in metrics_dict.keys():
            metrics_dict[metric_name] = metrics_dict[metric_name] / total_samples

        return metrics_dict


class Segmentation(NeuralNetTask):
    """Task implementation for binary and multiclass semantic segmentation.

    This class handles pixel-level classification tasks including binary and
    multiclass segmentation with customizable color mapping for visualization.

    Attributes:
        mode: Segmentation mode ("binary" or "multiclass").
        num_classes: Number of segmentation classes.
        threshold: Threshold for binary segmentation predictions.
        class_index_to_color: Dictionary mapping class indices to colors.
        palette: Color palette for visualization (PIL format).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        model_dir: str,
        mode: str = "binary",
        load_saved_model: bool = False,
        model_file_name: str = "latest_model.pt",
        device: str = "auto",
        num_classes: int = 1,
        threshold: float = 0.5,
        color_map: dict = None,
    ):
        """Initializes the Segmentation task.

        Args:
            model: PyTorch model architecture for segmentation.
            model_dir: Directory path for saving/loading model checkpoints.
            mode: Segmentation mode. Options: "binary" or "multiclass".
                Defaults to "binary".
            load_saved_model: Whether to load a previously saved model.
                Defaults to False.
            model_file_name: Name of the model checkpoint file.
                Defaults to "latest_model.pt".
            device: Device to use for computation. Options: "auto", "cpu",
                "cuda", or "mps". Defaults to "auto".
            num_classes: Number of segmentation classes. For binary
                segmentation, use 1 (class 0: background, class 1: foreground).
                Defaults to 1.
            threshold: Probability threshold for binary segmentation predictions.
                Defaults to 0.5.
            color_map: Dictionary mapping class indices to colors. If None,
                uses default color maps:
                - Binary: {0: 0, 1: 255} (grayscale)
                - Multiclass: {0: [0,0,0], 1: [R,G,B], ...} (RGB triplets)
                For multiclass, random RGB colors are generated if not specified.
                Class 0 is always background (black). Defaults to None.

        Raises:
            AssertionError: If num_classes is not an integer or is less than 1.

        Example:
            >>> model = UNet(in_channels=3, out_channels=3)
            >>> color_map = {0: [0,0,0], 1: [255,0,0], 2: [0,255,0]}
            >>> task = Segmentation(
            ...     model=model,
            ...     model_dir="./models",
            ...     mode="multiclass",
            ...     num_classes=3,
            ...     color_map=color_map
            ... )
        """

        super(Segmentation, self).__init__(
            model, model_dir, load_saved_model, model_file_name, device
        )
        assert isinstance(num_classes, int), "should be the number of classes"
        assert (
            num_classes >= 1
        ), "for segmentation task, it should be greater than 1 class"

        self.mode = mode
        self.num_classes = num_classes
        self.threshold = threshold

        if color_map:
            assert isinstance(color_map, dict)
            self.class_index_to_color = color_map
        else:
            if self.mode == "binary":
                self.class_index_to_color = {0: 0, 1: 255}
            else:
                self.class_index_to_color = {0: [0, 0, 0]}
                additional_colors = np.random.randint(
                    0, 256, size=(self.num_classes - 1, 3)
                )
                # Create random RGB color triplets
                for index, color in enumerate(additional_colors.tolist()):
                    self.class_index_to_color[index + 1] = color

        self.__create_color_palette()

    def __create_color_palette(self):
        """Creates a PIL-compatible color palette from the class color mapping.

        Generates a flat list of RGB values suitable for use with PIL Image
        palettes. For binary segmentation, creates a grayscale palette.
        For multiclass, converts the RGB color map to a flat array.

        Note:
            The palette is padded to 768 values (256 colors * 3 channels) as
            required by PIL.
        """
        if self.mode == "binary":
            self.palette = [0, 0, 0, 255, 255, 255]
        else:
            self.palette = (
                np.array(list(self.class_index_to_color.values()))
                .flatten()
                .astype(np.uint8)
                .tolist()
            )

        self.palette = self.palette + list(
            np.zeros(768 - (len(self.palette)), dtype=np.uint8).tolist()
        )

    def predict_batch(
        self, x: Union[torch.Tensor, np.ndarray], *args, **kwargs: Dict[str, Any]
    ) -> torch.Tensor:
        x = self.move_input_to_device(x, **kwargs)  # Move input to device

        if "model" in kwargs:
            pred = kwargs["model"](x)
        else:
            pred = self._model(x)

        if isinstance(pred, dict) and "out" in pred:
            return pred["out"]  # torchvision model's returns prediction in OrderedDict
        else:
            return pred

    def save_prediction(self, loader: torch.utils.data.DataLoader, save_dir: str):
        """Generates and saves segmentation predictions as PNG images.

        Performs inference on the data loader and saves predicted segmentation
        masks as PNG files with the appropriate color palette.

        Args:
            loader: DataLoader yielding batches of (images, filenames).
                The second element must be a list of filename strings.
            save_dir: Output directory path where prediction PNG files will be saved.
                Directory will be created if it doesn't exist.

        Raises:
            AssertionError: If loader is None, empty, or save_dir is None.

        Note:
            Filenames that don't end with '.png' will be automatically converted
            to PNG format with the .png extension.
        """
        assert loader is not None and len(loader) > 0
        assert save_dir is not None, "Output directory should not be none."

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        self._model = self._model.to(self._device)
        self._model.eval()
        with torch.no_grad():
            for x, y in tqdm(
                loader, total=len(loader), desc="{:12s}".format("Prediction")
            ):
                y_pred, x, y = self.eval_step(x, y)
                output_mask = self.transform_output(y_pred).cpu()
                self._save_image_batch(output_mask, y, save_dir)

    def _save_image_batch(
        self, class_indices: torch.Tensor, filenames: List[str], save_dir: str
    ):
        """Saves a batch of segmentation masks as PNG images.

        Args:
            class_indices: Batch of segmentation masks with shape (B, H, W)
                containing class indices.
            filenames: List of filenames for each image in the batch.
            save_dir: Directory path where images will be saved.

        Raises:
            AssertionError: If class_indices is not 3-dimensional (BHW format).

        Note:
            Images are saved as 8-bit palette PNG files with the task's color palette.
        """
        assert class_indices.ndim == 3, "should be in the form of BHW"
        for i in range(class_indices.shape[0]):
            image = Image.fromarray(
                class_indices[i].cpu().numpy().astype(np.uint8), "P"
            )
            image.putpalette(self.palette)
            filename = filenames[i]
            if not filename.endswith(".png"):
                filename = f"{filename.split('.')[0]}.png"
            image.save(os.path.join(save_dir, filename))

    def predict_class(self, loader: torch.utils.data.DataLoader):
        raise NotImplementedError()

    def show_predictions(
        self,
        loader: torch.utils.data.DataLoader,
        image_inverse_transform: Callable = None,
        samples: int = 4,
        cols: int = 3,
        figsize: Tuple[int, int] = (16, 16),
        target_known: bool = True,
    ):
        """Visualizes segmentation predictions on sample images.

        Displays input images, ground truth masks, and predicted masks in a
        matplotlib figure with overlays.

        Args:
            loader: DataLoader containing data for visualization.
            image_inverse_transform: Transformation to reverse image normalization
                for display. Defaults to None.
            samples: Number of samples to display. Defaults to 4.
            cols: Number of columns in the visualization grid. Defaults to 3.
            figsize: Figure size as (width, height) tuple. Defaults to (16, 16).
            target_known: Whether ground truth targets are available for comparison.
                Defaults to True.
        """
        self._model = self._model.to(self._device)
        self._model.eval()

        with torch.no_grad():
            x, targets = get_random_samples_batch_from_loader(loader, samples)
            predictions, x, targets = self.eval_step(x, targets).cpu()

            x = self.transform_input(x, image_inverse_transform)
            target_mask = self.decode_segmentation_mask(targets)
            class_indices = self.transform_output(predictions)
            output_mask = self.decode_segmentation_mask(class_indices)

            # BCHW --> #BHWC
            x = x.permute([0, 2, 3, 1])
            target_mask = target_mask.permute([0, 2, 3, 1])
            output_mask = output_mask.permute([0, 2, 3, 1])

            if self.num_classes == 1:
                target_mask = torch.cat([target_mask, target_mask, target_mask], dim=3)
                output_mask = torch.cat([output_mask, output_mask, output_mask], dim=3)

            images = []
            for i in range(x.shape[0]):
                images.extend([x[i], target_mask[i], output_mask[i]])

            image_titles = ["Input", "Target", "Prediction"] * x.shape[0]
            plot_images(images, image_titles, cols=cols, figsize=figsize, fontsize=12)

    def transform_target(self, y: torch.Tensor):
        """Transforms target mask to RGB color image for visualization.

        Args:
            y: Target segmentation mask with class indices.

        Returns:
            RGB color image tensor decoded using the class color palette.
        """
        return self.decode_segmentation_mask(y)

    def transform_output(self, predictions: torch.Tensor) -> torch.Tensor:
        """Converts model predictions to class indices.

        Applies sigmoid (binary) or softmax (multiclass) activation and converts
        probabilities to discrete class indices.

        Args:
            predictions: Model output logits of shape (B, C, H, W) where:
                - B: batch size
                - C: number of classes (1 for binary, >1 for multiclass)
                - H: height
                - W: width

        Returns:
            Tensor of class indices with shape (B, H, W). For binary segmentation,
            values are 0 or 1. For multiclass, values are in range [0, num_classes).

        Raises:
            AssertionError: If predictions is not 4-dimensional (BCHW format).

        Note:
            - Binary: Uses sigmoid activation with threshold (default 0.5)
            - Multiclass: Uses softmax activation with argmax
        """

        assert predictions.ndim == 4  # B,C,H,W

        if predictions.shape[1] == 1:
            # Binary
            probability = torch.sigmoid(predictions).squeeze(
                dim=1
            )  # TODO: handle for multilabel, squeeze not requried
            class_indices = torch.zeros_like(probability)
            class_indices[probability >= self.threshold] = 1
        else:
            # Multiclass
            probability = torch.softmax(predictions, dim=1)
            class_indices = torch.argmax(probability, dim=1)

        return class_indices

    def decode_segmentation_mask(self, class_indices: torch.Tensor) -> torch.Tensor:
        """Converts class indices to RGB color images for visualization.

        Args:
            class_indices: Batch of segmentation masks with shape (B, H, W)
                containing class indices.

        Returns:
            Batch of RGB images with shape (B, C, H, W) where:
                - For binary: C=1 (grayscale)
                - For multiclass: C=3 (RGB)
                Colors are mapped according to the class_index_to_color palette.

        Note:
            Uses PIL Image palette for efficient color mapping in multiclass mode.
        """
        # Convert to numpy array
        class_indices = class_indices.cpu().numpy().astype(np.uint8)

        decoded_images = []
        # For each image in the batch
        for i in range(class_indices.shape[0]):

            if self.mode == "binary":
                image_arr = np.zeros_like(class_indices[i])  # HW
                image_arr[class_indices[i] > 0] = 255
                image_arr = image_arr[np.newaxis, ...]  # CHW for grayscale
            else:
                image = Image.fromarray(class_indices[i], mode="P")
                image.putpalette(self.palette)
                image = image.convert("RGB")
                image_arr = np.array(image)  # HWC for RGB
                image_arr = image_arr.transpose(2, 0, 1)  # Convert to CHW format

            decoded_images.append(torch.from_numpy(image_arr))

        # return tensor of size (B, C, H, W) for both RGB and grayscale images
        return torch.stack(decoded_images)

    def log_prediction(
        self,
        tag: str,
        predictions: torch.Tensor,
        x: torch.Tensor,
        targets: torch.Tensor,
        logger: MLExperimentLogger,
        image_inverse_transform: Callable,
        global_step: int,
        img_size: Union[int, Tuple[int, int], None] = 224,
        **kwargs: dict,
    ):
        """Logs input images, target masks, and output masks to the experiment logger.

        Creates a visualization grid showing input images, ground truth masks,
        ground truth overlays, predicted masks, and predicted overlays side by side.

        Args:
            tag: Tag identifier for the logged images in the experiment tracker.
            predictions: Model predictions with shape (B, C, H, W) or (B, H, W).
            x: Input images with shape (B, C, H, W).
            targets: Ground truth masks with shape (B, H, W) or (B, C, H, W).
            logger: Experiment logger instance for tracking visualizations.
            image_inverse_transform: Callable to reverse image normalization
                for proper visualization.
            global_step: Current training step/epoch for the logger.
            img_size: Target size for resizing images. Can be int or (H, W) tuple.
                If None, no resizing is performed. Defaults to 224.
            **kwargs: Additional keyword arguments passed through.

        Note:
            Override this method to customize the logging behavior. The default
            implementation creates a grid with 5 images per sample: input, target
            mask, target overlay, predicted mask, and predicted overlay.
        """
        x = self.transform_input(x, image_inverse_transform).cpu()  # BCHW
        target_mask = self.decode_segmentation_mask(
            targets.cpu()
        )  # (B, C, H, W) for RGB images or (B, H, W) for grayscale images
        class_indices = self.transform_output(predictions).cpu()  # BHW
        output_mask = self.decode_segmentation_mask(
            class_indices
        )  # (B, C, H, W) for RGB images or (B, H, W) for grayscale images

        x = (x * 255.0).to(torch.uint8)  # Convert to uint8 for visualization

        target_segmentation = blend(x, target_mask)  # B, C, H, W
        output_segmentation = blend(x, output_mask)  # B, C, H, W

        # Resize images to img_size
        x = F.interpolate(x, size=img_size, mode="bilinear", align_corners=False)
        target_segmentation = F.interpolate(
            target_segmentation,
            size=img_size,
            mode="bilinear",
            align_corners=False,
        )
        target_mask = F.interpolate(
            target_mask,
            size=img_size,
            mode="bilinear",
            align_corners=False,
        )

        output_segmentation = F.interpolate(
            output_segmentation,
            size=img_size,
            mode="bilinear",
            align_corners=False,
        )

        output_mask = F.interpolate(
            output_mask,
            size=img_size,
            mode="bilinear",
            align_corners=False,
        )

        images = []
        for t in zip(
            x,
            target_mask,
            target_segmentation,
            output_mask,
            output_segmentation,
        ):
            images.extend(t)

        images = torch.stack(images)  # B * 5, C, H, W

        # nrow is number of images in a row, first is input, second is target mask, third is target mask overlay,
        # 4th is output mask, 5th is output mask overlay

        image_grid = torchvision.utils.make_grid(
            images, nrow=5, padding=5, pad_value=255
        )

        image_grid = image_grid.permute(1, 2, 0).cpu().numpy()  # CHW --> HWC

        logger.log_image(tag, image_grid, global_step)

    def write_prediction_to_logger(
        self,
        tag: str,
        loader: torch.utils.data.DataLoader,
        logger: MLExperimentLogger,
        image_inverse_transform: Callable,
        global_step: int,
        img_size: Union[int, Tuple[int, int], None] = 224,
        **kwargs: dict,
    ):
        """Writes input images, targets, and predictions to the experiment logger.

        Samples random batches from the data loader, generates predictions, and
        logs visualizations to the experiment tracker.

        Args:
            tag: Tag identifier for the logged images in the experiment tracker.
            loader: DataLoader containing data for visualization.
            logger: Experiment logger instance for tracking visualizations.
            image_inverse_transform: Callable to reverse image normalization
                for proper visualization.
            global_step: Current training step/epoch for the logger.
            img_size: Target size for resizing images. Can be int or (H, W) tuple.
                If None, no resizing is performed. Defaults to 224.
            **kwargs: Additional keyword arguments passed to eval_step.
        """

        self._model.eval()
        with torch.no_grad():
            x, targets = get_random_samples_batch_from_loader(loader, samples=4)
            predictions, x, targets = self.eval_step(x, targets, **kwargs)
            self.log_prediction(
                tag,
                predictions,
                x,
                targets,
                logger,
                image_inverse_transform,
                global_step,
                img_size,
                **kwargs,
            )


class ImageRegression(NeuralNetTask):
    """Task implementation for image regression problems.

    This class handles tasks where the model predicts continuous values from
    images, such as age estimation, pose estimation, or depth prediction.

    The task supports visualization of predictions alongside ground truth values
    and logging to experiment trackers.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        model_dir: str,
        load_saved_model: bool = False,
        model_file_name: str = "latest_model.pt",
        device: str = "auto",
    ):
        """Initializes the ImageRegression task.

        Args:
            model: PyTorch model instance for regression.
            model_dir: Directory path for saving and loading model checkpoints.
            load_saved_model: Whether to load a previously saved model from
                model_dir. Defaults to False.
            model_file_name: Name of the model checkpoint file.
                Defaults to "latest_model.pt".
            device: Device to use for computation. Options: "auto", "cpu",
                "cuda", or "mps". Defaults to "auto".
        """
        super(ImageRegression, self).__init__(
            model, model_dir, load_saved_model, model_file_name, device
        )

    def show_predictions(
        self,
        loader: torch.utils.data.DataLoader,
        image_inverse_transform: Callable = None,
        samples: int = 9,
        cols: int = 3,
        figsize: Tuple[int, int] = (10, 10),
        target_known: bool = True,
    ):
        """Visualizes model predictions on sample images.

        Displays random samples from the loader with their ground truth values
        and predicted values in a matplotlib figure.

        Args:
            loader: DataLoader containing data for visualization.
            image_inverse_transform: Transformation to reverse image normalization
                for display. Defaults to None.
            samples: Number of samples to display. Defaults to 9.
            cols: Number of columns in the visualization grid. Defaults to 3.
            figsize: Figure size as (width, height) tuple. Defaults to (10, 10).
            target_known: Whether ground truth targets are available for comparison.
                Defaults to True.
        """

        self._model = self._model.to(self._device)
        self._model.eval()

        with torch.no_grad():
            x, y = get_random_samples_batch_from_loader(loader, samples)
            predictions, x, y = self.eval_step(x, y)

            x = self.transform_input(x, image_inverse_transform)
            # #BCHW --> #BHWC
            x = x.permute([0, 2, 3, 1])

            def create_title(y, prediction):
                prediction = self.transform_output(prediction)
                if target_known:
                    return f"Ground Truth={self.transform_target(y)}\nPrediction={prediction}"
                else:
                    return f"{y}\nPrediction={prediction}"

            image_title_generator = (
                (x[index], create_title(y[index], predictions[index]), None)
                for index in range(x.shape[0])
            )

            plot_images_with_title(
                image_title_generator, samples=samples, cols=cols, figsize=figsize
            )

    def transform_target(self, y: torch.Tensor):
        """Transforms target tensor to a rounded float value.

        Args:
            y: Target tensor (single value).

        Returns:
            Rounded float value to 2 decimal places.
        """
        return round(y.item(), 2)

    def transform_output(self, prediction: torch.Tensor):
        """Transforms prediction tensor to a rounded float value.

        Args:
            prediction: Prediction tensor (single value).

        Returns:
            Rounded float value to 2 decimal places.
        """
        return round(prediction.item(), 2)

    def write_prediction_to_logger(
        self,
        tag: str,
        loader: torch.utils.data.DataLoader,
        logger: MLExperimentLogger,
        image_inverse_transform: Callable,
        global_step: int,
        img_size: Union[int, Tuple[int, int], None] = 224,
    ):
        """Writes predictions with ground truth values to the experiment logger.

        Creates a visualization grid showing input images alongside their
        ground truth and predicted values as text overlays.

        Args:
            tag: Unique tag identifier for the logged images.
            loader: DataLoader containing data for visualization.
            logger: Experiment logger instance for tracking visualizations.
            image_inverse_transform: Transformation to reverse image normalization.
            global_step: Current training epoch/step for the logger.
            img_size: Image size for TensorBoard logging. Can be int or (H, W) tuple.
                If None, no visualization is written. Defaults to 224.
        """

        if img_size:
            assert isinstance(img_size, int) or (
                isinstance(img_size, tuple) and len(img_size) == 2
            )
        else:
            return

        if isinstance(img_size, int):
            img_size = (img_size, img_size)

        self._model = self._model.to(self._device)
        self._model.eval()
        with torch.no_grad():
            x, targets = get_random_samples_batch_from_loader(loader)
            predictions, x, targets = self.eval_step(x, targets)

            x, y = x.cpu(), targets.cpu()
            x = self.transform_input(x, image_inverse_transform)
            input_img_size = tuple(x.shape[-2:])

            to_pillow_image = torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToPILImage(),
                    torchvision.transforms.Resize(img_size),
                ]
            )
            to_tensor = torchvision.transforms.ToTensor()

            text = "GT={ground_truth}\nPred={prediction}"
            output_images = []
            for index in range(x.shape[0]):
                ground_truth = self.transform_target(y[index])
                prediction = self.transform_output(predictions[index])
                content = text.format(ground_truth=ground_truth, prediction=prediction)
                content_image = create_text_image(content, img_size=img_size)

                if input_img_size != img_size:
                    output_images.append(
                        to_tensor(to_pillow_image(x[index].squeeze(dim=0)))
                    )
                else:
                    output_images.append(x[index].squeeze(dim=0))
                output_images.append(to_tensor(content_image))

            logger.log_artifact(f"{tag}", torch.stack(output_images), global_step)

    def predict_class(self, loader):
        raise NotImplementedError()


class ImageClassification(NeuralNetTask):
    """Task implementation for image classification.

    This class handles both binary and multiclass classification tasks where
    each image belongs to exactly one class. Supports custom class labels and
    visualization of predictions.

    Attributes:
        _classes: Optional sequence of class names for human-readable labels.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        model_dir: str,
        load_saved_model: bool = False,
        model_file_name: str = "latest_model.pt",
        device: str = "auto",
        classes: Sequence = None,
    ):
        """Initializes the ImageClassification task.

        Args:
            model: PyTorch model instance for classification.
            model_dir: Directory path for saving and loading model checkpoints.
            load_saved_model: Whether to load a previously saved model from
                model_dir. Defaults to False.
            model_file_name: Name of the model checkpoint file.
                Defaults to "latest_model.pt".
            device: Device to use for computation. Options: "auto", "cpu",
                "cuda", or "mps". Defaults to "auto".
            classes: Optional sequence of class names (e.g., ['cat', 'dog']).
                If provided, predictions will use these labels instead of
                class indices. Defaults to None.
        """
        super(ImageClassification, self).__init__(
            model, model_dir, load_saved_model, model_file_name, device
        )
        self._classes = classes

    def predict_class(self, loader: torch.utils.data.DataLoader):
        """Generates class predictions with probabilities for all data.

        Args:
            loader: DataLoader containing data for prediction.

        Returns:
            Tuple of (predicted_class, probability, targets) where:
                - predicted_class: Tensor of predicted class indices
                - probability: Tensor of prediction confidence scores
                - targets: Ground truth class labels
        """
        predictions, targets = self.predict(loader)
        predicted_class, probability = self.transform_output(predictions)
        return predicted_class, probability, targets

    def transform_target(self, y):
        """Transforms target class index to human-readable label if available.

        Args:
            y: Target class index.

        Returns:
            Class name if classes are defined, otherwise returns the index.
        """
        if self._classes:
            # if classes is not empty, replace target with actual class label
            y = self._classes[y]
        return y

    def transform_output(self, predictions: torch.Tensor):
        """Converts model predictions to class indices and probabilities.

        Applies sigmoid (binary) or softmax (multiclass) activation and extracts
        the predicted class and its probability.

        Args:
            predictions: Model output logits with shape (B, num_classes) for
                multiclass or (B, 1) for binary classification.

        Returns:
            Tuple of (indices, probabilities) where:
                - indices: Tensor of predicted class indices with shape (B,)
                - probabilities: Tensor of prediction confidences with shape (B,)

        Note:
            - Binary: Uses sigmoid with 0.5 threshold
            - Multiclass: Uses softmax with argmax
        """

        if predictions.shape[-1] > 1:
            # multiclass
            probability, indices = torch.max(F.softmax(predictions, dim=1), dim=1)
        else:
            # binary
            probability = torch.sigmoid(predictions)
            indices = torch.zeros_like(probability)
            indices[probability > 0.5] = 1

        return indices, probability

    def _create_title_for_display(
        self,
        target_class_index,
        predicted_class_index,
        predicted_probability,
        target_known=True,
    ):
        """Creates a colored title string for prediction visualization.

        Args:
            target_class_index: Ground truth class index.
            predicted_class_index: Predicted class index.
            predicted_probability: Prediction confidence score.
            target_known: Whether ground truth is available. Defaults to True.

        Returns:
            Tuple of (title_text, title_color) where:
                - title_text: Formatted string with prediction info
                - title_color: "green" (correct), "red" (incorrect), or "yellow" (unknown)
        """
        predicted_class = self.transform_target(predicted_class_index)
        probability = round(predicted_probability.item(), 2)
        if target_known:
            target_class = self.transform_target(target_class_index)
            title_color = "green" if predicted_class == target_class else "red"
            return (
                f"Ground Truth={target_class}"
                f"\nPrediction={predicted_class}, "
                f"{probability}",
                title_color,
            )
        else:
            return (
                f"{target_class_index}\nPrediction={predicted_class}, {probability}",
                "yellow",
            )

    def _create_output_image_for_tensorboard(
        self,
        target_class_index,
        predicted_class_index,
        predicted_probability,
        img_size=(224, 224),
    ):
        """Creates a text image showing prediction results for TensorBoard.

        Args:
            target_class_index: Ground truth class index.
            predicted_class_index: Predicted class index.
            predicted_probability: Prediction confidence score.
            img_size: Output image size as (height, width) tuple.
                Defaults to (224, 224).

        Returns:
            PIL Image containing formatted text with ground truth, prediction,
            and probability colored by correctness (green or red).
        """
        ground_truth = self.transform_target(target_class_index)
        predicted_class = self.transform_target(predicted_class_index)
        probability = round(predicted_probability.item(), 2)
        text_color = "green" if ground_truth == predicted_class else "red"
        display_content = f"{ground_truth}\n{predicted_class}, {probability}"
        return create_text_image(
            display_content, img_size=img_size, text_color=text_color
        )

    def show_predictions(
        self,
        loader: torch.utils.data.DataLoader,
        image_inverse_transform: Callable = None,
        samples: int = 9,
        cols: int = 3,
        figsize: Tuple[int, int] = (10, 10),
        target_known: bool = True,
    ):
        """Visualizes model predictions on sample images.

        Displays random samples from the loader with their ground truth labels,
        predicted labels, and confidence scores in a matplotlib figure.

        Args:
            loader: DataLoader containing data for visualization.
            image_inverse_transform: Transformation to reverse image normalization
                for display. Defaults to None.
            samples: Number of samples to display. Defaults to 9.
            cols: Number of columns in the visualization grid. Defaults to 3.
            figsize: Figure size as (width, height) tuple. Defaults to (10, 10).
            target_known: Whether ground truth targets are available for comparison.
                If True, titles will be colored green (correct) or red (incorrect).
                Defaults to True.
        """

        self._model = self._model.to(self._device)
        self._model.eval()

        with torch.no_grad():
            x, targets = get_random_samples_batch_from_loader(loader, samples)
            predictions, x, targets = self.eval_step(x, targets)

            x = self.transform_input(x, image_inverse_transform)
            # #BCHW --> #BHWC
            x = x.permute([0, 2, 3, 1])

            class_indices, probabilities = self.transform_output(predictions)

            image_title_generator = (
                (
                    x[index],
                    *self._create_title_for_display(
                        targets[index],
                        class_indices[index],
                        probabilities[index],
                        target_known,
                    ),
                )
                for index in range(x.shape[0])
            )

            plot_images_with_title(
                image_title_generator, samples=samples, cols=cols, figsize=figsize
            )

    def write_prediction_to_logger(
        self,
        tag: str,
        loader,
        logger: MLExperimentLogger,
        image_inverse_transform,
        global_step: int,
        img_size=224,
    ):
        """Writes predictions with labels to the experiment logger.

        Creates a visualization grid showing input images alongside their
        ground truth and predicted class labels with confidence scores.

        Args:
            tag: Unique tag identifier for the logged images.
            loader: DataLoader containing data for visualization.
            logger: Experiment logger instance for tracking visualizations.
            image_inverse_transform: Transformation to reverse image normalization.
            global_step: Current training epoch/step for the logger.
            img_size: Image size for logging. Can be int or (H, W) tuple.
                If None, no visualization is written. Defaults to 224.

        Note:
            Predictions are colored green for correct classifications and red
            for incorrect ones.
        """

        if img_size:
            assert isinstance(img_size, int) or (
                isinstance(img_size, tuple) and len(img_size) == 2
            )
        else:
            return

        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        self._model = self._model.to(self._device)
        self._model.eval()
        with torch.no_grad():
            x, targets = get_random_samples_batch_from_loader(loader)
            predictions, x, targets = self.eval_step(x, targets)

            x = self.transform_input(x).cpu()
            class_indices, probabilities = self.transform_output(predictions)

            input_img_size = tuple(x.shape[-2:])
            to_pillow_image = torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToPILImage(),
                    torchvision.transforms.Resize(img_size),
                ]
            )
            to_tensor = torchvision.transforms.ToTensor()

            output_images = []
            for index in range(x.shape[0]):
                content_image = self._create_output_image_for_tensorboard(
                    targets[index], class_indices[index], probabilities[index], img_size
                )
                if input_img_size != img_size:
                    output_images.append(
                        to_tensor(to_pillow_image(x[index].squeeze(dim=0)))
                    )
                else:
                    output_images.append(x[index].squeeze(dim=0))

                output_images.append(to_tensor(content_image))

            logger.log_artifact(f"{tag}", torch.stack(output_images), global_step)


class MultiLabelImageClassification(ImageClassification):
    """Task implementation for multi-label image classification.

    This class handles classification tasks where each image can belong to
    multiple classes simultaneously (e.g., an image containing both a cat
    and a dog).

    Attributes:
        _classes: Optional sequence of class names for human-readable labels.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        model_dir,
        load_saved_model: bool = False,
        model_file_name: str = "latest_model.pt",
        device: str = "auto",
        classes=None,
    ):
        """Initializes the MultiLabelImageClassification task.

        Args:
            model: PyTorch model instance for multi-label classification.
            model_dir: Directory path for saving and loading model checkpoints.
            load_saved_model: Whether to load a previously saved model from
                model_dir. Defaults to False.
            model_file_name: Name of the model checkpoint file.
                Defaults to "latest_model.pt".
            device: Device to use for computation. Options: "auto", "cpu",
                "cuda", or "mps". Defaults to "auto".
            classes: Optional sequence of class names for labeling.
                Defaults to None.
        """
        super(MultiLabelImageClassification, self).__init__(
            model, model_dir, load_saved_model, model_file_name, device
        )
        self._classes = classes

    def predict_class(self, loader):
        """Generates multi-label class predictions with probabilities for all data.

        Args:
            loader: DataLoader containing data for prediction.

        Returns:
            Tuple of (predicted_class, probability, targets) where:
                - predicted_class: Binary tensor indicating predicted classes
                - probability: Tensor of class probabilities for all classes
                - targets: Ground truth multi-label targets
        """
        predictions, targets = self.predict(loader)
        predicted_class, probability = self.transform_output(predictions)
        return predicted_class, probability, targets

    def transform_target(self, y):
        """Transforms target class indices to comma-separated class labels.

        Args:
            y: Binary tensor or list where 1 indicates the class is present.

        Returns:
            Comma-separated string of class names if classes are defined,
            otherwise returns the original indices.
        """
        if self._classes:
            # if classes is not empty, replace target with actual class label
            y = ", ".join(
                [self._classes[index] for index, value in enumerate(y) if value]
            )
        return y

    def transform_output(self, predictions):
        """Converts model predictions to binary class labels and probabilities.

        Applies sigmoid activation and thresholding to convert logits into
        multi-label predictions.

        Args:
            predictions: Model output logits with shape (B, num_classes).

        Returns:
            Tuple of (indices, probabilities) where:
                - indices: Binary tensor with shape (B, num_classes). Value is 1
                  if class is predicted (probability > 0.5), else 0.
                - probabilities: Tensor of class probabilities with shape
                  (B, num_classes) after sigmoid activation.

        Note:
            Uses sigmoid activation with 0.5 threshold for each class independently.
        """
        probability = torch.sigmoid(predictions)
        indices = torch.zeros_like(probability)
        indices[probability > 0.5] = 1

        return indices, probability

    def _create_title_for_display(
        self,
        target_class_indices,
        predicted_class_indexes,
        predicted_probs,
        target_known=True,
    ):
        """Creates a colored title string for multi-label prediction visualization.

        Args:
            target_class_indices: Binary tensor of ground truth class labels.
            predicted_class_indexes: Binary tensor of predicted class labels.
            predicted_probs: Tensor of prediction probabilities for each class.
            target_known: Whether ground truth is available. Defaults to True.

        Returns:
            Tuple of (title_text, title_color) where:
                - title_text: Formatted string with comma-separated class names
                  and their probabilities
                - title_color: "green" (correct), "red" (incorrect), or "yellow" (unknown)
        """
        predicted_classes = self.transform_target(predicted_class_indexes)
        predicted_probs = f'{", ".join([round(predicted_probs[prob_index], 2) for prob_index in predicted_class_indexes if prob_index])}'

        if target_known:
            target_classes = self.transform_target(target_class_indices)
            title_color = "green" if target_classes == predicted_classes else "red"
            return (
                f"GT={target_classes}\nPred={predicted_classes},\n{predicted_probs}",
                title_color,
            )
        else:
            return (
                f"{target_class_indices}\nPred={predicted_classes},\n{predicted_probs}",
                "yellow",
            )

    def _create_output_image_for_tensorboard(
        self,
        target_class_indices,
        predicted_class_indexes,
        predicted_probs,
        img_size=(224, 224),
    ):

        target_classes = self.transform_target(target_class_indices)
        predicted_classes = self.transform_target(predicted_class_indexes)
        probabilities = ", ".join(
            [
                round(predicted_probs[prob_index], 2)
                for prob_index in predicted_class_indexes
                if prob_index
            ]
        )
        display_content = f"{target_classes}\n{predicted_classes}\n{probabilities}"
        text_color = "green" if target_classes == predicted_classes else "red"
        return create_text_image(
            display_content, img_size=img_size, text_color=text_color
        )
