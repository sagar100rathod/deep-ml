import os
from abc import ABC, abstractmethod
from typing import Any, Optional

import torch
from torch.utils.tensorboard import SummaryWriter

from deepml import utils


class MLExperimentLogger(ABC):
    """Abstract base class for experiment tracking and logging.

    This class defines the interface for logging machine learning experiments
    across different platforms (TensorBoard, MLflow, Weights & Biases, etc.).

    Subclasses must implement all abstract methods to provide platform-specific
    logging functionality.
    """

    def __init__(self):
        """Initializes the MLExperimentLogger."""
        super(MLExperimentLogger, self).__init__()

    @abstractmethod
    def log_params(self, **kwargs):
        """Logs hyperparameters and configuration for the experiment.

        Args:
            **kwargs: Arbitrary keyword arguments containing parameters to log.
                Common parameters include model architecture, optimizer settings,
                learning rate, batch size, etc.
        """
        pass

    @abstractmethod
    def log_metric(self, tag: str, value: Any, step: int):
        """Logs a scalar metric value at a specific step.

        Args:
            tag: Identifier for the metric (e.g., "train/loss", "val/accuracy").
            value: Numeric value of the metric.
            step: Training step or epoch number for this metric value.
        """
        pass

    @abstractmethod
    def log_artifact(
        self, tag: str, value: Any, step: int, artifact_path: Optional[str] = None
    ):
        """Logs an artifact (file, tensor, or other data) to the experiment.

        Args:
            tag: Identifier for the artifact.
            value: The artifact data to log.
            step: Training step or epoch number.
            artifact_path: Optional file path for saving the artifact.
                Defaults to None.
        """
        pass

    @abstractmethod
    def log_model(
        self, tag: str, value: Any, step: int, artifact_path: Optional[str] = None
    ):
        """Logs a model checkpoint or weights to the experiment.

        Args:
            tag: Identifier for the model checkpoint.
            value: Model data or checkpoint information.
            step: Training step or epoch number.
            artifact_path: Optional file path to the model checkpoint.
                Defaults to None.
        """
        pass

    @abstractmethod
    def log_image(
        self, tag: str, value: Any, step: int, artifact_path: Optional[str] = None
    ):
        """Logs an image or batch of images to the experiment.

        Args:
            tag: Identifier for the image(s).
            value: Image data (tensor, numpy array, or PIL Image).
            step: Training step or epoch number.
            artifact_path: Optional file path for saving the image.
                Defaults to None.
        """
        pass


class TensorboardLogger(MLExperimentLogger):
    """TensorBoard experiment logger implementation.

    This logger writes experiment data to TensorBoard, including metrics,
    images, model graphs, and other artifacts.

    Attributes:
        writer: TensorBoard SummaryWriter instance for logging.
    """

    def __init__(self, model_dir):
        """Initializes the TensorboardLogger.

        Creates a new run directory within the model directory and initializes
        the TensorBoard SummaryWriter.

        Args:
            model_dir: Base directory path for saving TensorBoard logs.
                A new timestamped run directory will be created within this path.
        """
        super().__init__()
        self.__model_dir = model_dir
        self.writer = SummaryWriter(self.__model_dir)

    def log_params(self, **kwargs):
        """Logs hyperparameters and model graph to TensorBoard.

        Args:
            **kwargs: Keyword arguments. If 'task' and 'loader' are provided,
                writes the model computational graph to TensorBoard.
        """
        if "task" in kwargs and "loader" in kwargs:
            self.__write_graph_to_tensorboard(kwargs["task"], kwargs["loader"])
            self.writer.flush()

    def log_metric(self, tag: str, value: float, step: int):
        """Logs a scalar metric value to TensorBoard.

        Args:
            tag: Metric identifier (e.g., "train/loss", "val/accuracy").
            value: Numeric metric value.
            step: Training step or epoch number.
        """
        self.writer.add_scalar(tag, value, step)
        self.writer.flush()

    def log_artifact(
        self, tag: str, value: Any, step: int, artifact_path: Optional[str] = None
    ):
        """Logs an artifact to TensorBoard.

        Args:
            tag: Artifact identifier.
            value: Artifact data. If a torch.Tensor, logs as images.
            step: Training step or epoch number.
            artifact_path: Optional file path (unused in this implementation).
                Defaults to None.
        """

        if isinstance(value, torch.Tensor):
            self.writer.add_images(tag, torch.stack(value), step)
            self.writer.flush()

    def __write_graph_to_tensorboard(self, task, loader: torch.utils.data.DataLoader):
        """Writes the model computational graph to TensorBoard.

        Args:
            task: Task object containing the model.
            loader: DataLoader to extract a sample batch from.

        Note:
            Silently fails if graph writing is not supported by the model.
        """

        if not loader:
            # Write graph to tensorboard
            temp_x = None
            for X, _ in loader:
                temp_x = X
                break

            temp_x = task.move_input_to_device(temp_x)

            with torch.no_grad():
                task.model.eval()
                try:
                    self.writer.add_graph(task.model, temp_x)
                except Exception as e:
                    print("Warning: Failed to write graph to tensorboard.", e)

    def log_model(
        self, tag: str, value: Any, step: int, artifact_path: Optional[str] = None
    ):
        """Logs a model checkpoint to TensorBoard.

        Args:
            tag: Model identifier.
            value: Model data.
            step: Training step or epoch number.
            artifact_path: Optional file path to the model checkpoint.
                Defaults to None.
        """
        self.log_artifact(tag, value, step, artifact_path)

    def log_image(
        self, tag: str, value: Any, step: int, artifact_path: Optional[str] = None
    ):
        """Logs an image or batch of images to TensorBoard.

        Args:
            tag: Image identifier.
            value: Image data as a torch.Tensor with shape (B, C, H, W).
            step: Training step or epoch number.
            artifact_path: Optional file path (unused in this implementation).
                Defaults to None.

        Note:
            Only logs tensors with 4 dimensions (batch of images).
        """
        if isinstance(value, torch.Tensor) and value.ndim == 4:
            self.writer.add_images(tag, value, step)
            self.writer.flush()


class MLFlowLogger(MLExperimentLogger):
    """MLflow experiment logger implementation.

    This logger writes experiment data to MLflow tracking server, including
    metrics, parameters, model checkpoints, and images.

    Attributes:
        mlflow: MLflow module instance.
        log_model_weights: Whether to log model weights as artifacts.

    Note:
        Requires mlflow package to be installed.
    """

    try:
        import mlflow
    except ImportError as e:
        pass

    def __init__(
        self,
        experiment_name: str = "Default",
        tracking_uri: str = None,
        log_model_weights: bool = True,
    ):
        """Initializes the MLFlowLogger.

        Sets up the MLflow experiment and optionally configures the tracking URI.

        Args:
            experiment_name: Name of the MLflow experiment. Defaults to "Default".
            tracking_uri: URI of the MLflow tracking server. If None, uses the
                default local tracking. Defaults to None.
            log_model_weights: Whether to log model weights as artifacts.
                Defaults to True.
        """

        super().__init__()
        self.mlflow.set_experiment(experiment_name)
        self.log_model_weights = log_model_weights

        if tracking_uri:
            self.mlflow.set_tracking_uri(tracking_uri)

    def log_params(self, **kwargs):
        """Logs hyperparameters to MLflow.

        Args:
            **kwargs: Arbitrary keyword arguments containing parameters to log.
        """
        self.mlflow.log_params(kwargs)

    def log_metric(self, tag: str, value: Any, step: int):
        """Logs a scalar metric value to MLflow.

        Args:
            tag: Metric identifier (e.g., "train/loss", "val/accuracy").
            value: Numeric metric value.
            step: Training step or epoch number.
        """
        self.mlflow.log_metric(tag, value, step)

    def log_artifact(
        self, tag: str, value: Any, step: int, artifact_path: Optional[str] = None
    ):
        """Logs an artifact to MLflow.

        Args:
            tag: Artifact identifier.
            value: Artifact data.
            step: Training step or epoch number.
            artifact_path: Optional file path to the artifact. Defaults to None.

        Note:
            Currently not implemented. Override to add custom artifact logging.
        """
        pass

    def log_model(
        self, tag: str, value: Any, step: int, artifact_path: Optional[str] = None
    ):
        """Logs a model checkpoint to MLflow.

        Args:
            tag: Model identifier.
            value: Model data (unused).
            step: Training step or epoch number.
            artifact_path: File path to the model checkpoint.

        Note:
            Only logs if log_model_weights is True and artifact_path is provided.
        """
        if self.log_model_weights:
            self.mlflow.log_artifact(artifact_path, artifact_path=f"{tag}_epoch_{step}")

    def log_image(
        self, tag: str, value: Any, step: int, artifact_path: Optional[str] = None
    ):
        """Logs an image to MLflow.

        Args:
            tag: Image identifier/key.
            value: Image data as a numpy array or PIL Image.
            step: Training step or epoch number.
            artifact_path: Optional file path (unused). Defaults to None.
        """
        self.mlflow.log_image(value, key=tag, step=step)


class WandbLogger(MLExperimentLogger):
    """Weights & Biases (wandb) experiment logger implementation.

    This logger writes experiment data to Weights & Biases, including metrics,
    parameters, model artifacts, and images. Supports automatic cleanup of
    intermediate artifact versions to avoid storage overflow.

    Attributes:
        wandb: Wandb module instance.
        delete_intermediate_artifacts_versions: Whether to delete old artifact
            versions automatically.

    Note:
        Requires wandb package to be installed.
    """

    try:
        import wandb
    except ImportError as e:
        pass

    def __init__(
        self, delete_intermediate_artifacts_versions: bool = True, **kwargs: dict
    ):
        """Initializes the WandbLogger.

        Args:
            delete_intermediate_artifacts_versions: Whether to delete intermediate
                versions of artifacts during logging to avoid memory overflow.
                Defaults to True.
            **kwargs: Keyword arguments passed to wandb.init() for initialization.
                Common arguments include project, entity, name, config, etc.
        """
        super().__init__()
        self.delete_intermediate_artifacts_versions = (
            delete_intermediate_artifacts_versions
        )
        if kwargs:
            self.wandb.init(*kwargs)

    def log_params(self, **kwargs):
        """Logs hyperparameters to Weights & Biases.

        Args:
            **kwargs: Arbitrary keyword arguments containing parameters to log.
                These will be added to the wandb config.
        """
        self.wandb.config.update(kwargs, allow_val_change=True)

    def log_metric(self, tag: str, value: Any, step: int):
        """Logs a scalar metric value to Weights & Biases.

        Args:
            tag: Metric identifier (e.g., "train/loss", "val/accuracy").
            value: Numeric metric value.
            step: Training step or epoch number (unused, wandb auto-increments).
        """
        self.wandb.log({tag: value})

    def log_artifact(
        self, tag: str, value: Any, step: int, artifact_path: Optional[str] = None
    ):
        """Logs an artifact to Weights & Biases.

        Args:
            tag: Artifact identifier.
            value: Artifact data. If a 4D torch.Tensor, can be logged as images.
            step: Training step or epoch number.
            artifact_path: Optional file path to the artifact. Defaults to None.

        Note:
            Image logging for tensors is currently not implemented (TODO).
        """
        if isinstance(value, torch.Tensor) and value.ndim == 4:
            # TODO: log image
            pass

    def log_model(
        self, tag: str, value: Any, step: int, artifact_path: Optional[str] = None
    ):
        """Logs a model checkpoint to Weights & Biases.

        Creates a wandb Artifact for the model and optionally deletes older
        versions if delete_intermediate_artifacts_versions is True.

        Args:
            tag: Model identifier/artifact name.
            value: Model data (unused).
            step: Training step or epoch number (unused).
            artifact_path: File path to the model checkpoint file.

        Note:
            If delete_intermediate_artifacts_versions is enabled, only the
            latest version of the artifact is retained to save storage space.
        """
        if artifact_path and os.path.exists(artifact_path):
            artifact = self.wandb.Artifact(name=tag, type="model")
            artifact.add_file(local_path=artifact_path, name=tag)
            self.wandb.log_artifact(artifact)

            if self.delete_intermediate_artifacts_versions:

                # wait for properties to get populated for this artifact
                artifact.wait()
                latest_artifact_version = artifact.version

                for artifact in list(artifact.collection.artifacts()):
                    if artifact.version != latest_artifact_version:
                        artifact.delete()

    def log_image(
        self, tag: str, value: Any, step: int, artifact_path: Optional[str] = None
    ):
        """Logs an image to Weights & Biases.

        Args:
            tag: Image identifier/key for logging.
            value: Image data (numpy array, PIL Image, or tensor).
            step: Training step or epoch number (unused, wandb auto-increments).
            artifact_path: Optional file path (unused). Defaults to None.
        """
        self.wandb.log({tag: self.wandb.Image(value)})
