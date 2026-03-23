import os
from collections import OrderedDict, defaultdict
from typing import Callable, Dict, Optional, Tuple, Union

import torch
from accelerate import Accelerator
from tqdm import tqdm

from deepml.base import BaseLearner
from deepml.tasks import Task
from deepml.tracking import MLExperimentLogger, TensorboardLogger


class AcceleratorTrainer(BaseLearner):
    def __init__(
        self,
        task: Task,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        lr_scheduler_step_policy: str = "epoch",
        accelerator_config: Optional[dict] = None,
    ):

        super().__init__(
            task=task,
            optimizer=optimizer,
            criterion=criterion,
            lr_scheduler=lr_scheduler,
            lr_scheduler_fn=None,
            lr_scheduler_step_policy=lr_scheduler_step_policy,
        )

        if accelerator_config is None:
            accelerator_config = {}

        self.accelerator = Accelerator(**accelerator_config)
        self._task._device = self.accelerator.device
        self.epochs_completed = 0
        self.best_val_loss = float("inf")
        self.history = defaultdict(list)

        os.makedirs(self._model_dir, exist_ok=True)

    def __train(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        step_lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        metrics: Dict[str, torch.nn.Module] = None,
        non_blocking: bool = True,
        gradient_clip_value: Optional[float] = None,
        gradient_clip_max_norm: Optional[float] = None,
    ) -> OrderedDict[str, float]:

        model.train()

        # init all metrics with zeros each local process
        local_batch_metrics_dict = AcceleratorTrainer.init_metrics(metrics)

        training_progress_bar = None
        step = None
        global_metrics_dict = {}

        if self.accelerator.is_main_process:

            # Global metrics dict for tracking metrics from all processes,
            # separate history is used to track metrics across multiple calls to fit method
            global_metrics_dict = AcceleratorTrainer.init_metrics(metrics)
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

            with self.accelerator.accumulate(model):

                outputs, x, y = self._task.train_step(
                    x,
                    y,
                    model=model,
                    device=self.accelerator.device,
                    non_blocking=non_blocking,
                )

                if (
                    isinstance(outputs, torch.Tensor)
                    and outputs.ndim == 2
                    and outputs.shape[1] == 1
                ):
                    y = y.view_as(outputs)

                loss = criterion(outputs, y)
                self.accelerator.backward(
                    loss
                )  # no need to scale loss manually, accelerator takes care of it

                # Gradient clipping
                if self.accelerator.sync_gradients:
                    if gradient_clip_value is not None:
                        self.accelerator.clip_grad_value_(
                            model.parameters(), gradient_clip_value
                        )
                    elif gradient_clip_max_norm is not None:
                        self.accelerator.clip_grad_norm_(
                            model.parameters(), max_norm=gradient_clip_max_norm
                        )

                optimizer.step()

                if step_lr_scheduler is not None:
                    step_lr_scheduler.step()

                # Nullify the parameter gradients
                optimizer.zero_grad(set_to_none=True)

                local_batch_metrics_dict["loss"] = loss
                AcceleratorTrainer.update_metrics(
                    outputs, y, metrics, local_batch_metrics_dict
                )

                # collect metric values from all processes using tensor type, avoid dict type
                values = torch.tensor(
                    list(local_batch_metrics_dict.values()),
                    device=self.accelerator.device,
                    dtype=torch.float32,
                )

                # all_gather is used to aggregate the value across processes
                all_batch_metrics = self.accelerator.gather(
                    values
                )  # returns tensor of shape (world_size, num_metrics)

                # Aggregate metrics across all processes
                if self.accelerator.is_main_process:

                    training_progress_bar.update(1)

                    step = step + 1

                    all_batch_metrics = all_batch_metrics.view(
                        self.accelerator.num_processes, len(local_batch_metrics_dict)
                    )

                    # Convert all_batch_metrics to dict with metric names
                    all_batch_metrics = {
                        name: all_batch_metrics[
                            :, i
                        ]  # all_batch_metrics[:, 0] -> loss, all_batch_metrics[:, 1] -> acc, etc.
                        for i, name in enumerate(local_batch_metrics_dict.keys())
                    }

                    AcceleratorTrainer.update_metrics_with_simple_moving_average(
                        all_batch_metrics, global_metrics_dict, step
                    )
                    training_progress_bar.set_postfix(
                        {
                            name: f"{round(value, 4)}"
                            for name, value in global_metrics_dict.items()
                        }
                    )

        if self.accelerator.is_main_process:
            training_progress_bar.close()

        return global_metrics_dict

    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader = None,
        epochs: int = 10,
        save_model_after_every_epoch: int = 5,
        metrics: Dict[str, torch.nn.Module] = None,
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

        # Resume from checkpoint if provided
        if resume_from_checkpoint is not None and os.path.exists(
            resume_from_checkpoint
        ):
            state_dict = torch.load(
                resume_from_checkpoint, map_location=self.accelerator.device
            )
            self._model.load_state_dict(state_dict["model_state_dict"])

            if load_optimizer_state:
                AcceleratorTrainer.load_optimizer_state(self._optimizer, state_dict)

            if (
                self._lr_scheduler is not None
                and load_scheduler_state
                and "scheduler_state_dict" in state_dict
            ):
                AcceleratorTrainer.load_lr_schedular_state(
                    self._lr_scheduler, state_dict
                )

            self.epochs_completed = state_dict.get("epoch", 0)
            self.best_val_loss = state_dict.get("val_loss", float("inf"))

            if self.accelerator.is_main_process:
                print(
                    f"Resuming training from epoch {self.epochs_completed} with best validation loss {self.best_val_loss}"
                )

        # Prepare everything for the current device (CPU/GPU/TPU)
        model, optimizer, lr_scheduler, train_loader, val_loader = (
            self.accelerator.prepare(
                self._model,
                self._optimizer,
                self._lr_scheduler,
                train_loader,
                val_loader,
            )
        )

        if self.accelerator.is_main_process:
            self.logger = (
                logger if logger is not None else TensorboardLogger(self._model_dir)
            )
            self.logger.log_params(
                task=self._task,
                model=self._model,
                optimizer=self._optimizer,
                lr_scheduler=self._lr_scheduler,
                criterion=self._criterion,
                loader=val_loader,
                epochs=epochs,
                gradient_clip_value=gradient_clip_value,
                gradient_clip_max_norm=gradient_clip_max_norm,
                resume_from_checkpoint=resume_from_checkpoint,
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

            if self.accelerator.is_main_process:
                print("Epoch {}/{}:".format(epoch + 1, epochs))
                AcceleratorTrainer.write_lr(optimizer, epoch + 1, self.logger, history)

            # training
            train_global_metrics_dict = self.__train(
                model,
                optimizer,
                criterion,
                train_loader,
                step_lr_scheduler=(
                    lr_scheduler if self._lr_scheduler_step_policy == "step" else None
                ),
                metrics=metrics,
                non_blocking=non_blocking,
                gradient_clip_value=gradient_clip_value,
                gradient_clip_max_norm=gradient_clip_max_norm,
            )

            # evaluation
            if val_loader is not None:
                val_global_metrics_dict = self.__validate(
                    model,
                    val_loader,
                    criterion,
                    metrics,
                    non_blocking=non_blocking,
                )

            # After each epoch completed, write metrics to logger
            if self.accelerator.is_main_process:

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
                        self.accelerator.unwrap_model(model),
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
            self.accelerator.wait_for_everyone()

            # LR Scheduler step after each epoch
            if lr_scheduler is not None and self._lr_scheduler_step_policy == "epoch":
                if val_loader and isinstance(
                    lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    lr_scheduler.step(val_global_metrics_dict["loss"])
                else:
                    lr_scheduler.step()

        # Save latest model at the end
        if self.accelerator.is_main_process:
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

    @torch.no_grad()
    def __validate(
        self,
        model: torch.nn.Module,
        loader: torch.utils.data.DataLoader,
        criterion: torch.nn.Module,
        metrics: Dict[str, torch.nn.Module] = None,
        non_blocking: bool = True,
    ):

        model.eval()
        local_batch_metrics_dict = AcceleratorTrainer.init_metrics(metrics)
        validation_progress_bar = None
        global_metrics_dict = AcceleratorTrainer.init_metrics(metrics)
        step = 0

        if self.accelerator.is_main_process:
            validation_progress_bar = tqdm(
                total=len(loader),
                desc="{:12s}".format("Validation"),
                dynamic_ncols=True,
                leave=True,
            )

        for batch_index, (x, y) in enumerate(loader):

            outputs, x, y = self._task.eval_step(
                x,
                y,
                model=model,
                device=self.accelerator.device,
                non_blocking=non_blocking,
            )

            if isinstance(y, torch.Tensor):
                y = y.to(self.accelerator.device)

            if (
                isinstance(outputs, torch.Tensor)
                and outputs.ndim == 2
                and outputs.shape[1] == 1
            ):
                y = y.view_as(outputs)

            loss = criterion(outputs, y)

            local_batch_metrics_dict["loss"] = loss
            AcceleratorTrainer.update_metrics(
                outputs, y, metrics, local_batch_metrics_dict
            )

            # collect metric values from all processes using tensor type, avoid dict type
            values = torch.tensor(
                list(local_batch_metrics_dict.values()),
                device=self.accelerator.device,
                dtype=torch.float32,
            )

            # used to aggregate the value across processes
            all_batch_metrics = self.accelerator.gather(
                values
            )  # returns tensor of shape (world_size, num_metrics)

            # Aggregate metrics across all processes
            if self.accelerator.is_main_process:
                validation_progress_bar.update(1)
                step = step + 1

                all_batch_metrics = all_batch_metrics.view(
                    self.accelerator.num_processes, len(local_batch_metrics_dict)
                )

                # Convert all_batch_metrics to dict with metric names
                # all_batch_metrics[:, 0] -> loss, all_batch_metrics[:, 1] -> acc, etc.
                all_batch_metrics = {
                    name: all_batch_metrics[:, i]
                    for i, name in enumerate(local_batch_metrics_dict.keys())
                }

                AcceleratorTrainer.update_metrics_with_simple_moving_average(
                    all_batch_metrics, global_metrics_dict, step
                )
                validation_progress_bar.set_postfix(
                    {
                        name: f"{round(value, 4)}"
                        for name, value in global_metrics_dict.items()
                    }
                )

        if self.accelerator.is_main_process:
            validation_progress_bar.close()

        return global_metrics_dict

    def fit_temp(self, train_loader, val_loader, epochs=10, metrics: dict = {}):

        print("Number of processes:", self.accelerator.num_processes)
        print("Is main process:", self.accelerator.is_main_process)
        print("Device:", self.accelerator.device)

        # Prepare everything for the current device (CPU/GPU/TPU)
        model, optimizer, train_loader, val_loader = self.accelerator.prepare(
            self.model, self.optimizer, train_loader, val_loader
        )

        self.model = model
        self.optimizer = optimizer

        for epoch in range(epochs):

            # Important for multi-process shuffling
            if hasattr(train_loader, "sampler") and hasattr(
                train_loader.sampler, "set_epoch"
            ):
                train_loader.sampler.set_epoch(epoch)

            model.train()
            step = 0
            global_metrics = OrderedDict(loss=0.0, **{k: 0.0 for k in metrics})

            if self.accelerator.is_main_process:
                print(f"Epoch {epoch + 1}")
                pbar = tqdm(train_loader, desc="Training", dynamic_ncols=True)

            for batch in train_loader:
                inputs, targets = batch
                outputs = model(inputs)
                loss = self.criterion(outputs, targets)
                self.accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

                # Gather metrics across devices
                local_metrics = {"loss": loss.detach()}
                for name, metric_fn in metrics.items():
                    local_metrics[name] = metric_fn(outputs, targets)

                gathered = self.accelerator.gather_for_metrics(local_metrics)

                if self.accelerator.is_main_process:
                    step += 1
                    for k, v in gathered.items():
                        global_metrics[k] += (
                            v.mean().item() - global_metrics[k]
                        ) / step
                    pbar.set_postfix(
                        {f"train_{k}": round(v, 4) for k, v in global_metrics.items()}
                    )
                    pbar.update()

            if self.accelerator.is_main_process:
                pbar.close()

            # Validation loop
            model.eval()
            step = 0
            val_metrics = OrderedDict(loss=0.0, **{k: 0.0 for k in metrics})
            if self.accelerator.is_main_process:
                pbar = tqdm(val_loader, desc="Validation", dynamic_ncols=True)

            with torch.no_grad():
                for batch in val_loader:
                    inputs, targets = batch
                    outputs = model(inputs)
                    loss = self.criterion(outputs, targets)

                    local_metrics = {"loss": loss}
                    for name, metric_fn in metrics.items():
                        local_metrics[name] = metric_fn(outputs, targets)

                    gathered = self.accelerator.gather_for_metrics(local_metrics)

                    if self.accelerator.is_main_process:
                        step += 1
                        for k, v in gathered.items():
                            val_metrics[k] += (v.mean().item() - val_metrics[k]) / step
                        pbar.set_postfix(
                            {f"val_{k}": round(v, 4) for k, v in val_metrics.items()}
                        )
                        pbar.update()

            if self.accelerator.is_main_process:
                pbar.close()
                print("-" * 40)

        # Save model if needed
        if self.accelerator.is_main_process:
            self.accelerator.save_model(self.model, "./checkpoint")
            torch.save(
                {"optimizer": self.optimizer.state_dict()}, "./checkpoint/optimizer.pt"
            )
