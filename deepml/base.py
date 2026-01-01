import abc
import os
from collections import OrderedDict
from typing import Callable, Dict, Optional, Tuple, Union

import torch

from deepml.tasks import Task
from deepml.tracking import MLExperimentLogger


class BaseLearner(abc.ABC):
    def __init__(
        self,
        task: Task,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        lr_scheduler_fn: Optional[
            Callable[[torch.optim.Optimizer], torch.optim.lr_scheduler._LRScheduler]
        ] = None,
        lr_scheduler_step_policy: str = "epoch",
    ):

        assert isinstance(task, Task)
        assert (lr_scheduler is None) or (
            lr_scheduler_fn is None
        ), "Either lr_scheduler or lr_scheduler_fn can be provided, not both."

        self._task = task
        self._model = self._task.model
        self._model_dir = self._task.model_dir
        self._model_file_name = self._task.model_file_name
        self._optimizer = None
        self._criterion = None
        self._lr_scheduler = lr_scheduler
        self._lr_scheduler_fn = lr_scheduler_fn
        self._lr_scheduler_step_policy = None
        self.logger = None

        self.set_optimizer(optimizer)
        self.set_criterion(criterion)
        self.set_lr_scheduler_policy(lr_scheduler_step_policy)

    def set_optimizer(self, optimizer: torch.optim.Optimizer):
        assert isinstance(optimizer, torch.optim.Optimizer)
        self._optimizer = optimizer

    def set_criterion(self, criterion: torch.nn.Module):
        assert isinstance(criterion, torch.nn.Module)
        self._criterion = criterion

    def set_lr_scheduler_policy(self, lr_scheduler_step_policy: str = "epoch"):
        assert isinstance(
            lr_scheduler_step_policy, str
        ) and lr_scheduler_step_policy in ["epoch", "step"]
        self._lr_scheduler_step_policy = lr_scheduler_step_policy

    @staticmethod
    def load_optimizer_state(optimizer: torch.optim.Optimizer, state_dict: dict):
        if "optimizer" in state_dict and "optimizer_state_dict" in state_dict:
            if state_dict["optimizer"] == optimizer.__class__.__name__:
                optimizer.load_state_dict(state_dict["optimizer_state_dict"])
            else:
                print(
                    f"Skipping load optimizer state because {optimizer.__class__.__name__}"
                    f" != {state_dict['optimizer']}"
                )

    @staticmethod
    def load_lr_schedular_state(
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler, state_dict: dict
    ):
        if "scheduler" in state_dict and "scheduler_state_dict" in state_dict:
            if state_dict["scheduler"] == lr_scheduler.__class__.__name__:
                lr_scheduler.load_state_dict(state_dict["scheduler_state_dict"])
            else:
                print(
                    f"Skipping load lr scheduler state because {lr_scheduler.__class__.__name__}"
                    f" != {state_dict['scheduler']}"
                )

    def save(
        self,
        tag: str,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        epoch: int = -1,
        train_loss: float = float("inf"),
        val_loss: float = float("inf"),
    ):

        state_dict = {
            "model_state_dict": model.state_dict(),
            "optimizer": optimizer.__class__.__name__,
            "optimizer_state_dict": optimizer.state_dict(),
            "criterion": criterion.__class__.__name__,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        }

        if lr_scheduler is not None:
            state_dict["scheduler"] = lr_scheduler.__class__.__name__
            state_dict["scheduler_state_dict"] = lr_scheduler.state_dict()

        filepath = f"{os.path.join(self._model_dir, tag)}.pt"

        torch.save(state_dict, filepath)
        self.logger.log_model(tag, model, epoch, artifact_path=filepath)

        return filepath

    @staticmethod
    def init_metrics(metrics: Dict[str, torch.nn.Module]) -> OrderedDict[str, float]:
        metrics_dict = OrderedDict({"loss": 0.0})

        if metrics is None:
            return metrics_dict

        for metric_name, _ in metrics.items():
            if metric_name == "loss":
                raise ValueError("Metric name 'loss' is reserved of criterion")
            metrics_dict[metric_name] = 0.0

        return metrics_dict

    @staticmethod
    def update_metrics(
        outputs: torch.Tensor,
        targets: torch.Tensor,
        metrics_instance_dict: Dict[str, torch.nn.Module],
        target_metrics_dict: OrderedDict[str, float],
    ):

        if metrics_instance_dict is None:
            return

        for metric_name, metric_instance in metrics_instance_dict.items():
            target_metrics_dict[metric_name] = metric_instance(outputs, targets)

    @staticmethod
    def update_metrics_with_simple_moving_average(
        source_metrics_dict: Dict[str, torch.nn.Module],
        target_metrics_dict: OrderedDict[str, float],
        step: int,
    ):

        for metric_name, metric_value in source_metrics_dict.items():
            target_metrics_dict[metric_name] = target_metrics_dict[metric_name] + (
                metric_value.mean().item() - target_metrics_dict[metric_name]
            ) / float(step)

    @staticmethod
    def write_metrics_to_logger(
        metrics_dict: dict,
        tag: str,
        global_step: int,
        logger: MLExperimentLogger,
        history: dict,
    ):
        for name, value in metrics_dict.items():
            logger.log_metric(f"{name}/{tag}", value, global_step)
            history[f"{tag}_{name}"].append(value)

    @staticmethod
    def write_lr(
        optimizer, global_step: int, logger: MLExperimentLogger, history: dict
    ):
        # Write lr to tensor-board and history dict
        if len(optimizer.param_groups) == 1:
            param_group = optimizer.param_groups[0]
            logger.log_metric("learning_rate", param_group["lr"], global_step)
            history["learning_rate"].append(param_group["lr"])
        else:
            for index, param_group in enumerate(optimizer.param_groups):
                logger.log_metric(
                    f"learning_rate/param_group_{index}", param_group["lr"], global_step
                )
                history[f"learning_rate/param_group_{index}"].append(param_group["lr"])

    def log_metrics(
        self,
        val_loader: torch.utils.data.DataLoader,
        train_metrics: dict,
        val_metrics: dict,
        metrics_history: dict,
        epochs_completed: int,
        logger_img_size: Union[int, Tuple[int, int]],
        image_inverse_transform: Callable,
    ):

        BaseLearner.write_metrics_to_logger(
            train_metrics,
            "train",
            epochs_completed,
            self.logger,
            metrics_history,
        )

        BaseLearner.write_metrics_to_logger(
            val_metrics,
            "val",
            epochs_completed,
            self.logger,
            metrics_history,
        )

        # write random val images to tensorboard
        if logger_img_size is not None:
            self._task.write_prediction_to_logger(
                "val",
                val_loader,
                self.logger,
                image_inverse_transform,
                epochs_completed,
                img_size=logger_img_size,
            )

    def fit(self, *args, **kwargs):
        raise NotImplementedError("Subclass should implement this method.")

    def predict(self, *args, **kwargs):
        raise NotImplementedError("Subclass should implement this method.")
