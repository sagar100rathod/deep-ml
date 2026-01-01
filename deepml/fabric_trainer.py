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
        """
        Training class for learning a model weights using particular task.

        :param task: Object of subclass deepml.tasks.Task
        :param optimizer: The optimizer from torch.optim
        :param criterion: The loss function
        :param lr_scheduler_fn: Should be factory function returning desired learning rate scheduler, and accepting optimizer as an argument.
                                For example, lr_scheduler_fn = lambda optimizer: StepLR(optimizer, step_size=5, gamma=0.5)
        :param lr_scheduler_step_policy: It is the time when lr_scheduler.step() would be called.
                                         Possible choices are ["epoch", "step"]
                                         Default is "epoch" policy.
                                         Use "step" policy if you want lr_scheduler.step() to be
                                         called after each gradient step.
        :param accelerator: The hardware accelerator to use for training.
                         Possible choices - "cpu"``, ``"cuda"``, ``"mps"``, ``"gpu"``, ``"tpu"``, or ``"auto"``.
        :param strategy: Strategy for how to run across multiple devices.
                        Possible choices - "dp", "ddp", "fsdp", "deepspeed", "ddp_spawn" or "auto".
        :param devices: The number of devices to use for training.
        :param precision: The precision to use for training.
                        Possible choices are - "16-mixed", "32-true", "64-true", "bf16-mixed", "bf16-true" or "auto".

        :param num_nodes: The number of nodes to use for distributed training.
        :param fabric_plugins: Optional plugins to pass to Fabric for custom behaviors.
                               Like DeepSpeedPlugin for DeepSpeed integration.
                               BitsandbytesPlugin for memory efficient training with 8-bit optimizers etc.
                               Example:
                               from lightning_fabric.plugins import BitsandbytesPrecision
                               plugin = BitsandbytesPrecision(mode="int8")
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
        """
        Trains the model on specified train loader for specified number of epochs.

        Parameters
        ----------
        :param train_loader: The torch.utils.data.DataLoader for model to train on.

        :param val_loader: The torch.utils.data.DataLoader for model to validate on.
                           Default is None.

        :param epochs: int The number of epochs to train. Default is 10

        :param save_model_after_every_epoch: To save the model after every number of completed epochs
                                            Default is 5.

        :param metrics: dictionary of metrics 'metric_name': metric instance to monitor.
                        Metric name is used as label for logging metric value to console and tensorboard.
                        Metric instance must be a subclass of torch.nn.Module, which implements forward function and
                        returns calculated value.

        :param gradient_accumulation_steps : Number of steps to accumulate gradients before updating the model parameters.
                                    It is a way to simulate a larger batch size without increasing the memory footprint.

        :param gradient_clip_value: The maximum value for gradient clipping. Default is None which means no clipping.
                                    The gradients will be clipped to the range [-gradient_clip_value, gradient_clip_value]

        :param gradient_clip_max_norm:  Gradient clipping is done using the norm of the gradients.
                                        Default is None which means no clipping.

        :param resume_from_checkpoint: Full Path to the checkpoint file to resume training from.

        :param load_optimizer_state: If True, it will load optimizer state from checkpoint.

        :param load_scheduler_state: If True, it will load learning rate scheduler state from checkpoint.

        :param logger: MLExperimentLogger instance to log metrics and model artifacts.

        :param non_blocking:  weather to enable asynchronous cuda tensor transfer. Default is True.

        :param image_inverse_transform: It denotes reverse transformations of image normalization so that images
                                        can be displayed on tensor board.
                                        Default is deepml.transforms.ImageNetInverseTransform() which is
                                        an inverse of ImageNet normalization.

        :param logger_img_size:  image size to use for writing images to tensorboard

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
        """
        Trains the model on specified train loader for specified number of epochs using lightning_fabric.

        Parameters
        ----------
        :param train_loader: The torch.utils.data.DataLoader for model to train on.

        :param val_loader: The torch.utils.data.DataLoader for model to validate on.
                           Default is None.

        :param epochs: int The number of epochs to train. Default is 10

        :param save_model_after_every_epoch: To save the model after every number of completed epochs
                                            Default is 5.


        :param metrics: dictionary of metrics 'metric_name': metric instance to monitor.
                        Metric name is used as label for logging metric value to console and tensorboard.
                        Metric instance must be subclass of torch.nn.Module, which implements forward function and
                        returns calculated value.

        :param gradient_accumulation_steps : Number of steps to accumulate gradients before updating the model parameters.
                                    It is a way to simulate a larger batch size without increasing the memory footprint.

        :param gradient_clip_value: The maximum value for gradient clipping. Default is None which means no clipping.
                                       The gradients will be clipped to the range [-gradient_clip_value, gradient_clip_value]

        :param gradient_clip_max_norm:  Gradient clipping is done using the norm of the gradients.
                                        Default is None which means no clipping.

        :param resume_from_checkpoint: Full Path to the checkpoint file to resume training from.

        :param load_optimizer_state: If True, it will load optimizer state from checkpoint.

        :param load_scheduler_state: If True, it will load learning rate scheduler state from checkpoint.

        :param non_blocking:  weather to enable asynchronous cuda tensor transfer. Default is True.

        :param logger_img_size:  image size to use for writing images to tensorboard

        :param image_inverse_transform: It denotes reverse transformations of image normalization so that images
                                        can be displayed on tensor board.
                                        Default is deepml.transforms.ImageNetInverseTransform() which is
                                        an inverse of ImageNet normalization.

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

            if load_optimizer_state:
                FabricTrainer.load_optimizer_state(self._optimizer, state_dict)

            self.epochs_completed = state_dict.get("epoch", 0)
            self.best_val_loss = state_dict.get("val_loss", float("inf"))

            if fabric.is_global_zero:
                print(
                    f"Resuming training from epoch {self.epochs_completed} with best validation loss {self.best_val_loss}"
                )

        model, optimizer = fabric.setup(self._model, self._optimizer)
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
        """
        Run a single training epoch.

        Parameters
        ----------
        fabric : Fabric
            lightning_fabric Fabric instance used for device, sync and distributed utilities.
        model : torch.nn.Module
            Model to train; the function will set it to train() mode.
        optimizer : torch.optim.Optimizer
            Optimizer used to update model parameters.
        criterion : torch.nn.Module
            Loss function accepting (outputs, targets) and returning a scalar tensor.
        train_loader : torch.utils.data.DataLoader
            Iterable yielding batches in the form (inputs, targets).
        step_lr_scheduler : Optional[torch.optim.lr_scheduler._LRScheduler], optional
            Learning rate scheduler that should be stepped after each optimizer.step() (for \"step\" policy).
        metrics : Dict[str, torch.nn.Module], optional
            Mapping of metric name to metric module. Each metric should accept (outputs, targets).
        non_blocking : bool, optional
            If True, use non_blocking tensor transfers to device when available.
        gradient_accumulation_steps : int, optional
            Number of micro-batches to accumulate gradients over before calling optimizer.step().
        gradient_clip_value : Optional[float], optional
            If set, gradients will be clipped element-wise to the range [-gradient_clip_value, gradient_clip_value].
        gradient_clip_max_norm : Optional[float], optional
            If set, gradients will be clipped by global norm to this value.

        Returns
        -------
        OrderedDict[str, float]
            Aggregated metrics (simple moving average) across all processes. Only meaningful on the global zero process.

        Notes
        -----
        - Uses Fabric's `no_backward_sync` to avoid gradient sync during accumulation.
        - Aggregates per-batch metrics across processes using `fabric.all_gather` and computes a simple moving average.
        - Progress bars and returned metrics are managed only on the global zero process (`fabric.is_global_zero`).
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
        predictions, targets = self._task.predict(loader)
        return predictions, targets

    def predict_class(self, loader):
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

        self._task.show_predictions(
            loader,
            image_inverse_transform=image_inverse_transform,
            samples=samples,
            cols=cols,
            figsize=figsize,
            target_known=target_known,
        )
