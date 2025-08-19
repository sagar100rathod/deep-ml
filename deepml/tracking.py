import os
from abc import ABC, abstractmethod
from typing import Any, Optional

import torch
from torch.utils.tensorboard import SummaryWriter

from deepml import utils


class MLExperimentLogger(ABC):

    def __init__(self):
        super(MLExperimentLogger, self).__init__()

    @abstractmethod
    def log_params(self, **kwargs):
        pass

    @abstractmethod
    def log_metric(self, tag: str, value: Any, step: int):
        pass

    @abstractmethod
    def log_artifact(
        self, tag: str, value: Any, step: int, artifact_path: Optional[str] = None
    ):
        pass

    @abstractmethod
    def log_model(
        self, tag: str, value: Any, step: int, artifact_path: Optional[str] = None
    ):
        pass

    @abstractmethod
    def log_image(
        self, tag: str, value: Any, step: int, artifact_path: Optional[str] = None
    ):
        pass


class TensorboardLogger(MLExperimentLogger):

    def __init__(self, model_dir):
        super().__init__()
        self.__model_dir = model_dir
        self.writer = SummaryWriter(
            os.path.join(
                self.__model_dir, utils.find_new_run_dir_name(self.__model_dir)
            )
        )

    def log_params(self, **kwargs):
        if "task" in kwargs and "loader" in kwargs:
            self.__write_graph_to_tensorboard(kwargs["task"], kwargs["loader"])
            self.writer.flush()

    def log_metric(self, tag: str, value: float, step: int):
        self.writer.add_scalar(tag, value, step)
        self.writer.flush()

    def log_artifact(
        self, tag: str, value: Any, step: int, artifact_path: Optional[str] = None
    ):

        if isinstance(value, torch.Tensor):
            self.writer.add_images(tag, torch.stack(value), step)
            self.writer.flush()

    def __write_graph_to_tensorboard(self, task, loader: torch.utils.data.DataLoader):

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
        self.log_artifact(tag, value, step, artifact_path)

    def log_image(
        self, tag: str, value: Any, step: int, artifact_path: Optional[str] = None
    ):
        if isinstance(value, torch.Tensor) and value.ndim == 4:
            self.writer.add_images(tag, value, step)
            self.writer.flush()


class MLFlowLogger(MLExperimentLogger):

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
        """
        Initializes the MLFlowLogger with the specified experiment name and tracking URI.
        :param experiment_name:
        :param tracking_uri:
        :param log_model_weights: whether to log model weights as artifacts.
        """

        super().__init__()
        self.mlflow.set_experiment(experiment_name)
        self.log_model_weights = log_model_weights

        if tracking_uri:
            self.mlflow.set_tracking_uri(tracking_uri)

    def log_params(self, **kwargs):
        self.mlflow.log_params(kwargs)

    def log_metric(self, tag: str, value: Any, step: int):
        self.mlflow.log_metric(tag, value, step)

    def log_artifact(
        self, tag: str, value: Any, step: int, artifact_path: Optional[str] = None
    ):
        pass

    def log_model(
        self, tag: str, value: Any, step: int, artifact_path: Optional[str] = None
    ):
        if self.log_model_weights:
            self.mlflow.log_artifact(artifact_path, artifact_path=f"{tag}_epoch_{step}")

    def log_image(
        self, tag: str, value: Any, step: int, artifact_path: Optional[str] = None
    ):
        """
        Logs an image to MLFlow.
        :param tag: The tag for the image.
        :param value: The image data, can be a numpy array or a PIL Image.
        """
        self.mlflow.log_image(value, key=tag, step=step)


class WandbLogger(MLExperimentLogger):

    try:
        import wandb
    except ImportError as e:
        pass

    def __init__(
        self, delete_intermediate_artifacts_versions: bool = True, **kwargs: dict
    ):
        """
        :param delete_intermediate_artifacts_versions: Whether to delete intermediate versions of artifacts during logging, avoids memory overflow.
        :param kwargs: kwarg parameters for wandb.init
        """
        super().__init__()
        self.delete_intermediate_artifacts_versions = (
            delete_intermediate_artifacts_versions
        )
        if kwargs:
            self.wandb.init(*kwargs)

    def log_params(self, **kwargs):
        self.wandb.config.update(kwargs, allow_val_change=True)

    def log_metric(self, tag: str, value: Any, step: int):
        self.wandb.log({tag: value})

    def log_artifact(
        self, tag: str, value: Any, step: int, artifact_path: Optional[str] = None
    ):
        if isinstance(value, torch.Tensor) and value.ndim == 4:
            # TODO: log image
            pass

    def log_model(
        self, tag: str, value: Any, step: int, artifact_path: Optional[str] = None
    ):
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
        """
        Logs an image to Wandb.
        :param tag:
        :param value:
        :param step:
        :param artifact_path:
        :return:
        """
        self.wandb.log({tag: self.wandb.Image(value)}, step=step)
