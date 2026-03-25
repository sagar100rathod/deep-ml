import os
from collections import OrderedDict, defaultdict
from typing import Callable, Dict, Optional, Tuple, Union

import torch
from lightning_fabric import Fabric
from tqdm import tqdm

from deepml.base import BaseLearner
from deepml.tasks import Task
from deepml.tracking import MLExperimentLogger, TensorboardLogger


class FabricTrainer(BaseLearner):
    """Training class for learning model weights using Lightning Fabric.

    This trainer leverages Lightning Fabric for distributed training, mixed precision,
    and hardware acceleration while maintaining a simple PyTorch-like interface.

    It supports features like gradient accumulation, gradient clipping, learning rate
    scheduling, checkpointing, and logging with experiment tracking integration. The trainer
    is designed to be flexible and extensible for various types of learning tasks defined by the Task abstraction.

    """

    def __init__(
        self,
        task: Task,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        lr_scheduler_fn: Optional[
            Callable[[torch.optim.Optimizer], torch.optim.lr_scheduler._LRScheduler]
        ] = None,
        lr_scheduler_step_policy: str = "epoch",
        accelerator: Union[str, int] = "auto",
        strategy: Union[str, int] = "auto",
        devices: Union[str, int] = "auto",
        precision: str = "32-true",
        num_nodes: int = 1,
        fabric_plugins: Optional = None,
    ):
        """Initializes the FabricTrainer.

        Args:
            task: Task object defining the learning task (e.g., classification, segmentation).
            optimizer: PyTorch optimizer instance for parameter updates.
            criterion: Loss function module.
            lr_scheduler_fn: Factory function that creates a learning rate scheduler.
                Should accept an optimizer and return a scheduler instance.
                Example: ``lambda optimizer: StepLR(optimizer, step_size=5, gamma=0.5)``.
                Defaults to None.
            lr_scheduler_step_policy: When to call scheduler.step(). Valid options are
                ``"epoch"`` (step after each epoch) or ``"step"`` (step after each
                gradient update). Defaults to ``"epoch"``.
            accelerator: Hardware accelerator to use. Options: ``"cpu"``, ``"cuda"``,
                ``"mps"``, ``"gpu"``, ``"tpu"``, or ``"auto"``. Defaults to ``"auto"``.
            strategy: Distributed training strategy. Options: ``"dp"``, ``"ddp"``,
                ``"fsdp"``, ``"deepspeed"``, ``"ddp_spawn"``, or ``"auto"``.
                Defaults to ``"auto"``.
            devices: Number or list of devices to use. Can be int, str, or ``"auto"``.
                Defaults to ``"auto"``.
            precision: Training precision. Options: ``"16-mixed"``, ``"32-true"``,
                ``"64-true"``, ``"bf16-mixed"``, ``"bf16-true"``, or ``"auto"``.
                Defaults to ``"32-true"``.
            num_nodes: Number of nodes for multi-node distributed training.
                Defaults to 1.
            fabric_plugins: Optional Fabric plugins for custom behaviors (e.g.,
                DeepSpeedPlugin, BitsandbytesPrecision). Defaults to None.

        Example:
            >>> from lightning_fabric.plugins import BitsandbytesPrecision
            >>> plugin = BitsandbytesPrecision(mode="int8")
            >>> trainer = FabricTrainer(
            ...     task=task,
            ...     optimizer=optimizer,
            ...     criterion=criterion,
            ...     fabric_plugins=plugin
            ... )
        """
        super().__init__(
            task=task,
            optimizer=optimizer,
            criterion=criterion,
            lr_scheduler=None,
            lr_scheduler_fn=lr_scheduler_fn,
            lr_scheduler_step_policy=lr_scheduler_step_policy,
        )

        self.fabric = Fabric(
            accelerator=accelerator,
            strategy=strategy,
            devices=devices,
            precision=precision,
            num_nodes=num_nodes,
            plugins=fabric_plugins,
        )

        self._task._device = self.fabric.device
        self.epochs_completed = 0
        self.best_val_loss = float("inf")
        self.history = defaultdict(list)
        self.logger = None

        os.makedirs(self._model_dir, exist_ok=True)

    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader = None,
        epochs: int = 10,
        save_model_after_every_epoch: int = 5,
        metrics: Dict[str, torch.nn.Module] = None,
        gradient_accumulation_steps: int = 1,
        gradient_clip_value: Optional[float] = None,
        gradient_clip_max_norm: Optional[float] = None,
        resume_from_checkpoint: str = None,
        load_optimizer_state: bool = False,
        load_scheduler_state: bool = False,
        logger: MLExperimentLogger = None,
        non_blocking: bool = True,
        image_inverse_transform: Callable = None,
        logger_img_size: Union[int, Tuple[int, int]] = None,
    ):
        """Trains the model for the specified number of epochs.

        This method launches distributed training using Lightning Fabric and handles
        checkpointing, logging, and training history management.

        Args:
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data. Defaults to None.
            epochs: Total number of epochs to train. Defaults to 10.
            save_model_after_every_epoch: Frequency (in epochs) to save model checkpoints.
                Defaults to 5.
            metrics: Dictionary mapping metric names to metric instances. Each metric
                must be a torch.nn.Module with a forward() method. Defaults to None.
            gradient_accumulation_steps: Number of steps to accumulate gradients before
                performing an optimizer step. Simulates larger batch sizes. Defaults to 1.
            gradient_clip_value: Maximum absolute value for gradient clipping. Gradients
                will be clipped to [-gradient_clip_value, gradient_clip_value].
                Defaults to None (no clipping).
            gradient_clip_max_norm: Maximum L2 norm for gradient clipping. Defaults to
                None (no clipping).
            resume_from_checkpoint: Path to checkpoint file to resume training from.
                Defaults to None.
            load_optimizer_state: Whether to load optimizer state from checkpoint.
                Defaults to False.
            load_scheduler_state: Whether to load learning rate scheduler state from
                checkpoint. Defaults to False.
            logger: Experiment logger for tracking metrics and artifacts. If None, uses
                TensorboardLogger. Defaults to None.
            non_blocking: Whether to use asynchronous CUDA tensor transfers.
                Defaults to True.
            image_inverse_transform: Transformation to reverse image normalization for
                visualization in TensorBoard. Defaults to None.
            logger_img_size: Image size (int or tuple) for TensorBoard logging.
                Defaults to None.

        Note:
            After training completes, the latest model checkpoint is automatically loaded
            into the trainer's model and optimizer.
        """

        history = self.fabric.launch(
            self._fit_impl,
            train_loader,
            val_loader=val_loader,
            epochs=epochs,
            save_model_after_every_epoch=save_model_after_every_epoch,
            metrics=metrics,
            gradient_accumulation_steps=gradient_accumulation_steps,
            gradient_clip_value=gradient_clip_value,
            gradient_clip_max_norm=gradient_clip_max_norm,
            resume_from_checkpoint=resume_from_checkpoint,
            load_optimizer_state=load_optimizer_state,
            load_scheduler_state=load_scheduler_state,
            logger=logger,
            non_blocking=non_blocking,
            image_inverse_transform=image_inverse_transform,
            logger_img_size=logger_img_size,
        )

        # after training is complete, load model weights back
        if self.fabric.is_global_zero:
            latest_checkpoint_filepath = (
                f"{os.path.join(self._model_dir, 'latest_model')}.pt"
            )
            state_dict = torch.load(
                latest_checkpoint_filepath, map_location=self.fabric.device
            )
            self._model.load_state_dict(state_dict["model_state_dict"])
            self._optimizer.load_state_dict(state_dict["optimizer_state_dict"])
            self.epochs_completed = state_dict.get("epoch", 0)
            self.best_val_loss = state_dict.get("val_loss", float("inf"))

        # update history list
        for key, value in history.items():
            self.history[key].extend(value)

    def _fit_impl(
        self,
        fabric: Fabric,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader = None,
        epochs: int = 10,
        save_model_after_every_epoch: int = 5,
        metrics: Dict[str, torch.nn.Module] = None,
        gradient_accumulation_steps: int = 1,
        gradient_clip_value: Optional[float] = None,
        gradient_clip_max_norm: Optional[float] = None,
        resume_from_checkpoint: str = None,
        load_optimizer_state: bool = False,
        load_scheduler_state: bool = False,
        logger: MLExperimentLogger = None,
        non_blocking: bool = True,
        image_inverse_transform: Callable = None,
        logger_img_size: Union[int, Tuple[int, int]] = None,
    ) -> Dict[str, list]:
        """Internal implementation of training loop using Lightning Fabric.

        This method is launched by Fabric and runs the actual training loop across
        distributed processes. It handles model setup, checkpointing, validation,
        and metric tracking.

        Args:
            fabric: Lightning Fabric instance for distributed training utilities.
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data. Defaults to None.
            epochs: Total number of epochs to train. Defaults to 10.
            save_model_after_every_epoch: Frequency (in epochs) to save model checkpoints.
                Defaults to 5.
            metrics: Dictionary mapping metric names to metric instances. Each metric
                must be a torch.nn.Module with a forward() method. Defaults to None.
            gradient_accumulation_steps: Number of steps to accumulate gradients before
                performing an optimizer step. Must be greater than 0. Defaults to 1.
            gradient_clip_value: Maximum absolute value for gradient clipping. Gradients
                will be clipped to [-gradient_clip_value, gradient_clip_value].
                Defaults to None (no clipping).
            gradient_clip_max_norm: Maximum L2 norm for gradient clipping. Defaults to
                None (no clipping).
            resume_from_checkpoint: Path to checkpoint file to resume training from.
                Defaults to None.
            load_optimizer_state: Whether to load optimizer state from checkpoint.
                Defaults to False.
            load_scheduler_state: Whether to load learning rate scheduler state from
                checkpoint. Defaults to False.
            logger: Experiment logger for tracking metrics and artifacts. Defaults to None.
            non_blocking: Whether to use asynchronous CUDA tensor transfers.
                Defaults to True.
            image_inverse_transform: Transformation to reverse image normalization for
                visualization in TensorBoard. Defaults to None.
            logger_img_size: Image size (int or tuple) for TensorBoard logging.
                Defaults to None.

        Returns:
            Dictionary containing training history with metric names as keys and
            lists of values as entries.

        Raises:
            AssertionError: If gradient_accumulation_steps is not greater than 0.
            ValueError: If both gradient_clip_value and gradient_clip_max_norm are
                provided (only one can be used).
            TypeError: If any metric is not a torch.nn.Module with a forward() method.

        Note:
            Only the global zero process saves checkpoints and manages the logger.
            All processes synchronize at the end of each epoch using fabric.barrier().
        """

        assert (
            gradient_accumulation_steps > 0
        ), "Accumulation steps should be greater than 0"

        if gradient_clip_value is not None and gradient_clip_max_norm is not None:
            raise ValueError(
                "Only one of gradient_clip_value or gradient_clip_max_norm should be passed."
            )

        # Check valid metrics types
        if metrics:
            for metric_name, metric_instance in metrics.items():
                if not (
                    isinstance(metric_instance, torch.nn.Module)
                    and hasattr(metric_instance, "forward")
                ):
                    raise TypeError(f"{metric_instance.__class__} is not supported")

        state_dict = {}
        if resume_from_checkpoint is not None and os.path.exists(
            resume_from_checkpoint
        ):
            state_dict = torch.load(resume_from_checkpoint, map_location=fabric.device)
            self._model.load_state_dict(state_dict["model_state_dict"])

            self.epochs_completed = state_dict.get("epoch", 0)
            self.best_val_loss = state_dict.get("val_loss", float("inf"))

            if fabric.is_global_zero:
                print(
                    f"Resuming training from epoch {self.epochs_completed} with best validation loss {self.best_val_loss}"
                )

        model, optimizer = fabric.setup(self._model, self._optimizer)

        if load_optimizer_state:
            FabricTrainer.load_optimizer_state(optimizer, state_dict)

        train_loader, val_loader = fabric.setup_dataloaders(train_loader, val_loader)
        lr_scheduler = (
            self._lr_scheduler_fn(optimizer)
            if self._lr_scheduler_fn is not None
            else None
        )

        if (
            lr_scheduler is not None
            and load_scheduler_state
            and "scheduler_state_dict" in state_dict
        ):
            FabricTrainer.load_lr_schedular_state(lr_scheduler, state_dict)

        if fabric.is_global_zero:
            self.logger = (
                logger if logger is not None else TensorboardLogger(self._model_dir)
            )
            self.logger.log_params(
                task=self._task,
                loader=val_loader,
                epochs=epochs,
                criterion=self._criterion,
                lr_scheduler=lr_scheduler,
            )

        criterion = self._criterion
        epochs_completed = self.epochs_completed
        best_val_loss = self.best_val_loss
        epochs = epochs_completed + epochs
        history = defaultdict(list)

        val_global_metrics_dict = {"loss": float("inf")}

        train_loss = float("inf")
        val_loss = float("inf")

        for epoch in range(epochs_completed, epochs):

            if fabric.is_global_zero:
                print("Epoch {}/{}:".format(epoch + 1, epochs))
                FabricTrainer.write_lr(optimizer, epoch + 1, self.logger, history)

            # training
            train_global_metrics_dict = self.__train(
                fabric,
                model,
                optimizer,
                criterion,
                train_loader,
                step_lr_scheduler=(
                    lr_scheduler if self._lr_scheduler_step_policy == "step" else None
                ),
                metrics=metrics,
                non_blocking=non_blocking,
                gradient_accumulation_steps=gradient_accumulation_steps,
                gradient_clip_value=gradient_clip_value,
                gradient_clip_max_norm=gradient_clip_max_norm,
            )

            # evaluation
            if val_loader is not None:
                val_global_metrics_dict = self.__validate(
                    fabric,
                    model,
                    val_loader,
                    criterion,
                    metrics,
                    non_blocking=non_blocking,
                )

            # After each epoch completed, write metrics to logger
            if fabric.is_global_zero:

                epochs_completed = epochs_completed + 1
                self.log_metrics(
                    val_loader,
                    train_global_metrics_dict,
                    val_global_metrics_dict,
                    history,
                    epochs_completed,
                    logger_img_size,
                    image_inverse_transform,
                )

                train_loss = train_global_metrics_dict["loss"]
                val_loss = val_global_metrics_dict["loss"]

                message = f"\nTrain Loss: {train_loss:.4f} Val Loss: {val_loss:.4f}"

                # Save the best validation model
                if val_loss < best_val_loss:
                    message = message + " [Saving best validation model]"
                    best_val_loss = val_loss
                    self.save(
                        "best_val_model",
                        model,
                        optimizer,
                        criterion,
                        lr_scheduler,
                        epoch=epochs_completed,
                        train_loss=train_loss,
                        val_loss=val_loss,
                    )

                # Log info message to console only global zero process
                tqdm.write(message)

                if epochs_completed % save_model_after_every_epoch == 0:
                    last_checkpoint = "epoch_{}_model".format(epochs_completed)
                    self.save(
                        last_checkpoint,
                        model,
                        optimizer,
                        criterion,
                        lr_scheduler,
                        epoch=epochs_completed,
                        train_loss=train_loss,
                        val_loss=val_loss,
                    )

            # Ensure all processes are synchronized before proceeding next epoch
            fabric.barrier()

            # LR Scheduler step after each epoch
            if lr_scheduler is not None and self._lr_scheduler_step_policy == "epoch":
                if val_loader and isinstance(
                    lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    lr_scheduler.step(val_global_metrics_dict["loss"])
                else:
                    lr_scheduler.step()

        # Save latest model at the end
        if fabric.is_global_zero:
            self.save(
                "latest_model",
                model,
                optimizer,
                criterion,
                lr_scheduler,
                epoch=epochs_completed,
                train_loss=train_loss,
                val_loss=val_loss,
            )

        return history

    def __train(
        self,
        fabric: Fabric,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        step_lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        metrics: Dict[str, torch.nn.Module] = None,
        non_blocking: bool = True,
        gradient_accumulation_steps: int = 1,
        gradient_clip_value: Optional[float] = None,
        gradient_clip_max_norm: Optional[float] = None,
    ) -> OrderedDict[str, float]:
        """Runs a single training epoch with gradient accumulation and distributed training support.

        Args:
            fabric: Lightning Fabric instance used for device, sync and distributed utilities.
            model: Model to train; the function will set it to train() mode.
            optimizer: Optimizer used to update model parameters.
            criterion: Loss function accepting (outputs, targets) and returning a scalar tensor.
            train_loader: Iterable yielding batches in the form (inputs, targets).
            step_lr_scheduler: Learning rate scheduler that should be stepped after each
                optimizer.step() (for "step" policy). Defaults to None.
            metrics: Mapping of metric name to metric module. Each metric should accept
                (outputs, targets). Defaults to None.
            non_blocking: If True, use non_blocking tensor transfers to device when available.
                Defaults to True.
            gradient_accumulation_steps: Number of micro-batches to accumulate gradients over
                before calling optimizer.step(). Defaults to 1.
            gradient_clip_value: If set, gradients will be clipped element-wise to the range
                [-gradient_clip_value, gradient_clip_value]. Defaults to None.
            gradient_clip_max_norm: If set, gradients will be clipped by global norm to this
                value. Defaults to None.

        Returns:
            OrderedDict mapping metric names to aggregated values (simple moving average)
            across all processes. Only meaningful on the global zero process.

        Note:
            - Uses Fabric's ``no_backward_sync`` to avoid gradient sync during accumulation.
            - Aggregates per-batch metrics across processes using ``fabric.all_gather`` and
              computes a simple moving average.
            - Progress bars and returned metrics are managed only on the global zero process.
        """

        # Training mode
        model.train()

        # init all metrics with zeros
        local_batch_metrics_dict = FabricTrainer.init_metrics(metrics)

        training_progress_bar = None
        step = None
        global_metrics_dict = {}

        if fabric.is_global_zero:

            # Global metrics dict for tracking metrics from all processes,
            # separate history is used to track metrics across multiple calls to fit method
            global_metrics_dict = FabricTrainer.init_metrics(metrics)
            training_progress_bar = tqdm(
                total=len(train_loader),
                desc="{:12s}".format("Training"),
                dynamic_ncols=True,
            )
            # count number of steps
            step = 0

        # Nullify the parameter gradients
        optimizer.zero_grad(set_to_none=True)

        for batch_index, (x, y) in enumerate(train_loader):

            is_accumulating = batch_index % gradient_accumulation_steps != 0
            is_last_batch = (batch_index + 1) == len(train_loader)

            # If we are accumulating gradients, we do not need to step the optimizer
            with fabric.no_backward_sync(model, enabled=is_accumulating):
                outputs, x, y = self._task.train_step(
                    x, y, model=model, device=fabric.device, non_blocking=non_blocking
                )

                if (
                    isinstance(outputs, torch.Tensor)
                    and outputs.ndim == 2
                    and outputs.shape[1] == 1
                ):
                    y = y.view_as(outputs)

                loss = criterion(outputs, y)
                fabric.backward(loss / gradient_accumulation_steps)  # normalize loss

            # we gather log loss and metrics at each batch, so no need to sum up running loss during accumulation
            local_batch_metrics_dict["loss"] = loss
            FabricTrainer.update_metrics(outputs, y, metrics, local_batch_metrics_dict)

            # collect metric values from all processes using tensor type, avoid dict type
            values = torch.tensor(
                list(local_batch_metrics_dict.values()),
                device=fabric.device,
                dtype=torch.float32,
            )

            # all_gather is used to aggregate the value across processes
            all_batch_metrics = fabric.all_gather(
                values
            )  # returns tensor of shape (world_size, num_metrics)

            # update progress bar for each batch
            # Aggregate metrics across all processes
            if fabric.is_global_zero:
                training_progress_bar.update(1)

                step = step + 1

                all_batch_metrics = all_batch_metrics.view(
                    fabric.world_size, len(local_batch_metrics_dict)
                )

                # Convert all_batch_metrics to dict with metric names
                all_batch_metrics = {
                    name: all_batch_metrics[
                        :, i
                    ]  # all_batch_metrics[:, 0] -> loss, all_batch_metrics[:, 1] -> acc, etc.
                    for i, name in enumerate(local_batch_metrics_dict.keys())
                }

                FabricTrainer.update_metrics_with_simple_moving_average(
                    all_batch_metrics, global_metrics_dict, step
                )
                training_progress_bar.set_postfix(
                    {
                        name: f"{round(value, 4)}"
                        for name, value in global_metrics_dict.items()
                    }
                )

            # If we are not accumulating gradients, we step the optimizer
            if not is_accumulating or is_last_batch:

                # Gradient clipping
                if gradient_clip_value is not None:
                    fabric.clip_gradients(
                        model, optimizer, clip_val=gradient_clip_value
                    )
                elif gradient_clip_max_norm is not None:
                    fabric.clip_gradients(
                        model, optimizer, max_norm=gradient_clip_max_norm
                    )

                optimizer.step()

                if step_lr_scheduler is not None:
                    step_lr_scheduler.step()

                # Nullify the parameter gradients
                optimizer.zero_grad(set_to_none=True)

        if fabric.is_global_zero:
            training_progress_bar.close()

        return global_metrics_dict

    @torch.no_grad()
    def __validate(
        self,
        fabric: Fabric,
        model: torch.nn.Module,
        loader: torch.utils.data.DataLoader,
        criterion: torch.nn.Module,
        metrics: Dict[str, torch.nn.Module] = None,
        non_blocking: bool = True,
    ):
        """Runs a single validation epoch across all processes.

        Args:
            fabric: Lightning Fabric instance used for device, sync and distributed utilities.
            model: Model to evaluate; the function will set it to eval() mode.
            loader: DataLoader yielding batches in the form (inputs, targets).
            criterion: Loss function accepting (outputs, targets) and returning a scalar tensor.
            metrics: Mapping of metric name to metric module. Each metric should accept
                (outputs, targets). Defaults to None.
            non_blocking: If True, use non_blocking tensor transfers to device when available.
                Defaults to True.

        Returns:
            OrderedDict mapping metric names to aggregated values (simple moving average)
            across all processes. Only meaningful on the global zero process.

        Note:
            - Gradients are disabled via ``@torch.no_grad()`` decorator.
            - Aggregates per-batch metrics across processes using ``fabric.all_gather``.
            - Progress bars and returned metrics are managed only on the global zero process.
        """

        model.eval()
        local_batch_metrics_dict = FabricTrainer.init_metrics(metrics)
        validation_progress_bar = None
        global_metrics_dict = FabricTrainer.init_metrics(metrics)
        step = 0

        if fabric.is_global_zero:
            validation_progress_bar = tqdm(
                total=len(loader),
                desc="{:12s}".format("Validation"),
                dynamic_ncols=True,
                leave=True,
            )

        for batch_index, (x, y) in enumerate(loader):

            outputs, x, y = self._task.eval_step(
                x, y, model=model, device=fabric.device, non_blocking=non_blocking
            )

            if isinstance(y, torch.Tensor):
                y = y.to(fabric.device)

            if (
                isinstance(outputs, torch.Tensor)
                and outputs.ndim == 2
                and outputs.shape[1] == 1
            ):
                y = y.view_as(outputs)

            loss = criterion(outputs, y)

            local_batch_metrics_dict["loss"] = loss
            FabricTrainer.update_metrics(outputs, y, metrics, local_batch_metrics_dict)

            # collect metric values from all processes using tensor type, avoid dict type
            values = torch.tensor(
                list(local_batch_metrics_dict.values()),
                device=fabric.device,
                dtype=torch.float32,
            )

            # all_gather is used to aggregate the value across processes
            all_batch_metrics = fabric.all_gather(
                values
            )  # returns tensor of shape (world_size, num_metrics)

            # Aggregate metrics across all processes
            if fabric.is_global_zero:
                validation_progress_bar.update(1)
                step = step + 1

                all_batch_metrics = all_batch_metrics.view(
                    fabric.world_size, len(local_batch_metrics_dict)
                )

                # Convert all_batch_metrics to dict with metric names
                # all_batch_metrics[:, 0] -> loss, all_batch_metrics[:, 1] -> acc, etc.
                all_batch_metrics = {
                    name: all_batch_metrics[:, i]
                    for i, name in enumerate(local_batch_metrics_dict.keys())
                }

                FabricTrainer.update_metrics_with_simple_moving_average(
                    all_batch_metrics, global_metrics_dict, step
                )
                validation_progress_bar.set_postfix(
                    {
                        name: f"{round(value, 4)}"
                        for name, value in global_metrics_dict.items()
                    }
                )

        if fabric.is_global_zero:
            validation_progress_bar.close()

        return global_metrics_dict

    def predict(self, loader):
        """Generates predictions for the given data loader.

        Args:
            loader: DataLoader containing data for prediction.

        Returns:
            Tuple of (predictions, targets) where predictions are the model outputs
            and targets are the ground truth labels.
        """
        predictions, targets = self._task.predict(loader)
        return predictions, targets

    def predict_class(self, loader):
        """Generates class predictions with probabilities for the given data loader.

        Args:
            loader: DataLoader containing data for prediction.

        Returns:
            Tuple of (predicted_class, probability, targets) where:
                - predicted_class: Predicted class labels
                - probability: Class probabilities or confidence scores
                - targets: Ground truth labels
        """
        predicted_class, probability, targets = self._task.predict_class(loader)
        return predicted_class, probability, targets

    def show_predictions(
        self,
        loader,
        image_inverse_transform=None,
        samples=9,
        cols=3,
        figsize=(10, 10),
        target_known=True,
    ):
        """Visualizes model predictions on sample images.

        Args:
            loader: DataLoader containing data for visualization.
            image_inverse_transform: Transformation to reverse image normalization for
                display. Defaults to None.
            samples: Number of samples to display. Defaults to 9.
            cols: Number of columns in the visualization grid. Defaults to 3.
            figsize: Figure size as (width, height) tuple. Defaults to (10, 10).
            target_known: Whether ground truth targets are available for comparison.
                Defaults to True.
        """

        self._task.show_predictions(
            loader,
            image_inverse_transform=image_inverse_transform,
            samples=samples,
            cols=cols,
            figsize=figsize,
            target_known=target_known,
        )
