import os
import csv
from collections import OrderedDict, defaultdict
from typing import Tuple, Callable, Union, Dict, Optional

import numpy as np
import torch
from tqdm import tqdm
from lightning_fabric import Fabric
from torch.utils.tensorboard import SummaryWriter

import deepml.tasks
from deepml.tasks import Task
from deepml.tracking import MLExperimentLogger, TensorboardLogger


class Learner:

    def __init__(self, task: Task,
                 optimizer: torch.optim.Optimizer,
                 criterion: torch.nn.Module,
                 accelerator: str = "auto",
                 strategy: str ="auto",
                 devices: str="auto",
                 precision: str ="32-true",
                 lr_scheduler_fn: Optional[Callable[[torch.optim.Optimizer], torch.optim.lr_scheduler._LRScheduler]] = None,
                 lr_scheduler_step_policy: str = "epoch",
                 logger: MLExperimentLogger = None,
                 num_nodes: int = 1):
        """
        Training class for learning a model weights using particular task.

        :param task: Object of subclass deepml.tasks.Task
        :param optimizer: The optimizer from torch.optim
        :param criterion: The loss function
        :param accelerator: The hardware accelerator to use for training.
                         Possible choices - "cpu"``, ``"cuda"``, ``"mps"``, ``"gpu"``, ``"tpu"``, or ``"auto"``.
        :param strategy: Strategy for how to run across multiple devices.
                        Possible choices - "dp", "ddp", "fsdp", "deepspeed", "ddp_spawn" or "auto".
        :param devices: The number of devices to use for training.
        :param precision: The precision to use for training.
                        Possible choices are - "16-mixed", "32-true", "64-true", "bf16-mixed", "bf16-true" or "auto".

        :param lr_scheduler_fn: Should be factory function returning desired learning rate scheduler, and accepting optimizer as an argument.
                                For example, lr_scheduler_fn = lambda optimizer: StepLR(optimizer, step_size=5, gamma=0.5)
        :param lr_scheduler_step_policy: It is the time when lr_scheduler.step() would be called.
                                         Possible choices are ["epoch", "step"]
                                         Default is "epoch" policy.
                                         Use "step" policy if you want lr_scheduler.step() to be
                                         called after each gradient step.
        :param num_nodes: The number of nodes to use for distributed training.
        """
        assert isinstance(task, Task)

        self.__predictor = task
        self.__model = self.__predictor.model
        self.__model_dir = self.__predictor.model_dir
        self.__model_file_name = self.__predictor.model_file_name
        self.__device = self.__predictor.device
        self.__optimizer = None
        self.__criterion = None
        self.__lr_scheduler_fn = lr_scheduler_fn

        assert lr_scheduler_step_policy in ["epoch", "step"], "lr_scheduler_step_policy should be either 'epoch' or 'step'"
        self.__lr_scheduler_step_policy = lr_scheduler_step_policy

        self.set_optimizer(optimizer)
        self.set_criterion(criterion)

        self.fabric = Fabric(accelerator=accelerator, strategy=strategy,
                             devices=devices, precision=precision, num_nodes=num_nodes)

        self.epochs_completed = 0
        self.best_val_loss = float("inf")
        self.history = defaultdict(list)
        self.logger = logger

        if self.logger is None:
            os.makedirs(self.__model_dir, exist_ok=True)
            self.logger = TensorboardLogger(self.__model_dir)

    def set_optimizer(self, optimizer: torch.optim.Optimizer):
        assert isinstance(optimizer, torch.optim.Optimizer)
        self.__optimizer = optimizer

    def set_criterion(self, criterion: torch.nn.Module):
        assert isinstance(criterion, torch.nn.Module)
        self.__criterion = criterion

    def set_lr_scheduler(self, lr_scheduler_fn, lr_scheduler_step_policy: str = "epoch"):
        if lr_scheduler_fn is not None:
            self.__lr_scheduler_fn = lr_scheduler_fn

        assert isinstance(lr_scheduler_step_policy, str) and lr_scheduler_step_policy in ['epoch', 'batch']
        self.__lr_scheduler_step_policy = lr_scheduler_step_policy

    @staticmethod
    def __load_optimizer_state(optimizer: torch.optim.Optimizer, state_dict: dict):
        if 'optimizer' in state_dict and 'optimizer_state_dict' in state_dict:
            if state_dict['optimizer'] == optimizer.__class__.__name__:
                optimizer.load_state_dict(state_dict['optimizer_state_dict'])
            else:
                print(f"Skipping load optimizer state because {optimizer.__class__.__name__}"
                      f" != {state_dict['optimizer']}")

    @staticmethod
    def __load_lr_schedular_state(lr_scheduler: torch.optim.lr_scheduler._LRScheduler, state_dict: dict):
        if 'scheduler' in state_dict and 'scheduler_state_dict' in state_dict:
            if state_dict['scheduler'] == lr_scheduler.__class__.__name__:
                lr_scheduler.load_state_dict(state_dict['scheduler_state_dict'])
            else:
                print(f"Skipping load lr scheduler state because {lr_scheduler.__class__.__name__}"
                      f" != {state_dict['scheduler']}")

    def save(self, tag: str, model: torch.nn.Module,
             optimizer: torch.optim.Optimizer,
             criterion: torch.nn.Module,
             lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
             epoch: int = -1,
             train_loss: float = float("inf"),
             val_loss: float = float("inf")):

        state_dict = {'model_state_dict': model.state_dict(),
                     'optimizer': optimizer.__class__.__name__,
                     'optimizer_state_dict': optimizer.state_dict(),
                     'criterion': criterion.__class__.__name__,
                     'epoch': epoch,
                      "train_loss": train_loss,
                      "val_loss": val_loss}

        if lr_scheduler is not None:
            state_dict['scheduler'] = lr_scheduler.__class__.__name__
            state_dict['scheduler_state_dict'] = lr_scheduler.state_dict()

        filepath = f"{os.path.join(self.__model_dir, tag)}.pt"

        torch.save(state_dict, filepath)
        self.logger.log_model(tag, model, epoch, artifact_path=filepath)

        return filepath

    def set_predictor(self, predictor: deepml.tasks.Task):
        assert isinstance(predictor, Task)
        self.__predictor = predictor

    @staticmethod
    def __init_metrics(metrics: Dict[str, torch.nn.Module]) -> OrderedDict[str, float]:
        metrics_dict = OrderedDict({'loss': 0})

        if metrics is None:
            return metrics_dict

        for metric_name, _ in metrics.items():
            if metric_name == "loss":
                raise ValueError("Metric name 'loss' is reserved of criterion")
            metrics_dict[metric_name] = 0

        return metrics_dict

    @staticmethod
    def __update_metrics(outputs: torch.Tensor, targets: torch.Tensor,
                         metrics_instance_dict: Dict[str, torch.nn.Module],
                         target_metrics_dict: OrderedDict[str, float]):

        if metrics_instance_dict is None:
            return

        for metric_name, metric_instance in metrics_instance_dict.items():
            target_metrics_dict[metric_name] = metric_instance(outputs, targets)

    @staticmethod
    def __update_metrics_with_simple_moving_average(source_metrics_dict: Dict[str, torch.nn.Module],
                         target_metrics_dict: OrderedDict[str, float], step: int):

        for metric_name, metric_value in source_metrics_dict.items():
            target_metrics_dict[metric_name] = target_metrics_dict[metric_name] + (metric_value.mean().item() -
                                                                                   target_metrics_dict[metric_name]) / step

    @staticmethod
    def __write_metrics_to_logger(metrics_dict: dict, tag: str, global_step: int, logger: MLExperimentLogger,
                                  history: dict):
        for name, value in metrics_dict.items():
            logger.log_metric(f'{name}/{tag}', value, global_step)
            history[f"{tag}_{name}"].append(value)


    @staticmethod
    def __write_lr(optimizer, global_step: int, logger: MLExperimentLogger, history: dict):
        # Write lr to tensor-board and history dict
        if len(optimizer.param_groups) == 1:
            param_group = optimizer.param_groups[0]
            logger.log_metric('learning_rate', param_group['lr'], global_step)
            history['learning_rate'].append(param_group['lr'])
        else:
            for index, param_group in enumerate(optimizer.param_groups):
                logger.log_metric(f'learning_rate/param_group_{index}', param_group['lr'],
                                       global_step)
                history[f'learning_rate/param_group_{index}'].append(param_group['lr'])


    def fit(self,
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
            non_blocking: bool = True,
            image_inverse_transform: Callable = None,
            logger_img_size: Union[int, Tuple[int, int]] = None):
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

            :param non_blocking:  weather to enable asynchronous cuda tensor transfer. Default is True.

            :param image_inverse_transform: It denotes reverse transformations of image normalization so that images
                                           can be displayed on tensor board.
                                           Default is deepml.transforms.ImageNetInverseTransform() which is
                                           an inverse of ImageNet normalization.

            :param logger_img_size:  image size to use for writing images to tensorboard

         """

        history = self.fabric.launch(self._fit_impl,
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
                                    non_blocking=non_blocking,
                                    image_inverse_transform=image_inverse_transform,
                                    logger_img_size=logger_img_size)

        # after training is complete, load model weights back
        latest_checkpoint_filepath = f"{os.path.join(self.__model_dir, 'latest_model')}.pt"

        state_dict = torch.load(latest_checkpoint_filepath, map_location="cpu")
        self.__model.load_state_dict(state_dict['model_state_dict'])
        self.__optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        self.epochs_completed = state_dict.get("epoch", 0)
        self.best_val_loss = state_dict.get('val_loss', float("inf"))

        # updater history list
        for key, value in history.items():
            self.history[key].extend(value)


    def _fit_impl(self, fabric: Fabric, train_loader: torch.utils.data.DataLoader,
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

        assert gradient_accumulation_steps > 0, "Accumulation steps should be greater than 0"

        if gradient_clip_value is not None and gradient_clip_max_norm is not None:
            raise ValueError("Only one of gradient_clip_value or gradient_clip_max_norm should be passed.")

        # Check valid metrics types
        if metrics:
            for metric_name, metric_instance in metrics.items():
                if not (isinstance(metric_instance, torch.nn.Module) and hasattr(metric_instance, 'forward')):
                    raise TypeError(f'{metric_instance.__class__} is not supported')

        state_dict = {}
        if resume_from_checkpoint is not None and os.path.exists(resume_from_checkpoint):
            state_dict = torch.load(resume_from_checkpoint, map_location="cpu")
            self.__model.load_state_dict(state_dict['model_state_dict'])

            if load_optimizer_state:
                Learner.__load_optimizer_state(self.__optimizer, state_dict)

            self.epochs_completed = state_dict.get("epoch", 0)
            self.best_val_loss = state_dict.get('val_loss', float("inf"))

            print(f"Resuming training from epoch {self.epochs_completed} with best validation loss {self.best_val_loss}")

        model, optimizer = fabric.setup(self.__model, self.__optimizer)
        train_loader, val_loader = fabric.setup_dataloaders(train_loader, val_loader)
        lr_scheduler = self.__lr_scheduler_fn(optimizer) if self.__lr_scheduler_fn is not None else None

        if lr_scheduler is not None and load_scheduler_state and 'scheduler_state_dict' in state_dict:
            Learner.__load_lr_schedular_state(lr_scheduler, state_dict)

        model.to(fabric.device)
        optimizer.to(fabric.device)
        criterion = self.__criterion.to(fabric.device)
        epochs_completed = self.epochs_completed
        best_val_loss = self.best_val_loss
        epochs = epochs_completed + epochs
        history = defaultdict(list)

        val_global_metrics_dict = {"loss": float("inf")}

        train_loss = float("inf")
        val_loss = float("inf")

        for epoch in range(epochs_completed, epochs):

            if fabric.is_global_zero:
                Learner.__write_lr(optimizer, epoch, self.logger, history)

            # training
            train_global_metrics_dict = self.__train(fabric,
                                                     model,
                                                     optimizer,
                                                    criterion, train_loader,
                                                    step_lr_scheduler=lr_scheduler if self.__lr_scheduler_step_policy == "step" else None,
                                                    metrics=metrics,
                                                    non_blocking=non_blocking,
                                                    gradient_accumulation_steps=gradient_accumulation_steps,
                                                    gradient_clip_value=gradient_clip_value,
                                                    gradient_clip_algorithm=gradient_clip_algorithm)

            # evaluation
            if val_loader is not None:
                val_global_metrics_dict = self.__validate(fabric, model, val_loader, criterion, metrics,
                                                          non_blocking=non_blocking)

            train_loss = train_global_metrics_dict['loss']
            val_loss =  val_global_metrics_dict['loss']

            # After each epoch completed, write metrics to logger
            if fabric.is_global_zero:
                epochs_completed = epochs_completed + 1

                Learner.__write_metrics_to_logger(train_global_metrics_dict,'train', epochs_completed,
                                                self.logger, history)

                Learner.__write_metrics_to_logger(val_global_metrics_dict,'val', epochs_completed,
                                                self.logger, history)

                message = f"Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f}"

                # Save best validation model
                if val_loss < best_val_loss:
                    message = message + " [Saving best validation model]"
                    best_val_loss = val_loss
                    self.save('best_val_model', model, optimizer, criterion, lr_scheduler, epoch=epochs_completed,
                              train_loss=train_loss, val_loss=val_loss)

                # Log info message to console only global zero process
                print(message)
                print("-" * 40)

            # LR Scheduler step after each epoch
            if lr_scheduler is not None and self.__lr_scheduler_step_policy == "epoch":
                if val_loader and isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    lr_scheduler.step(val_global_metrics_dict["loss"])
                else:
                    lr_scheduler.step()

            # Save model checkpoints after every save_model_after_every_epoch
            if fabric.is_global_zero and epochs_completed % save_model_after_every_epoch == 0:
                last_checkpoint = "epoch_{}_model".format(epochs_completed)
                self.save(last_checkpoint, model, optimizer, criterion, lr_scheduler, epoch=epochs_completed,
                          train_loss=train_loss, val_loss=val_loss)


        # Save latest model at the end
        if fabric.is_global_zero:
            self.save("latest_model", model, optimizer, criterion, lr_scheduler, epoch=epochs_completed,
                      train_loss=train_loss, val_loss=val_loss)

        return history


    def __train(self,
                fabric: Fabric,
                model: torch.nn.Module,
                optimizer: torch.optim.Optimizer,
                criterion: torch.nn.Module,
                train_loader : torch.utils.data.DataLoader,
                step_lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                metrics: Dict[str, torch.nn.Module] = None,
                non_blocking: bool = True,
                gradient_accumulation_steps: int = 1,
                gradient_clip_value: Optional[float] = None,
                gradient_clip_max_norm: Optional[float] = None) -> OrderedDict[str, float]:

        assert model is not None
        assert isinstance(train_loader, torch.utils.data.DataLoader) and len(train_loader) > 0

        # Training mode
        model.train()

        # init all metrics with zeros
        local_batch_metrics_dict = Learner.__init_metrics(metrics)

        training_progress_bar = None
        step = None
        global_metrics_dict = {}

        if fabric.is_global_zero:

            # Global metrics dict for tracking metrics from all processes,
            # separate history is used to track metrics across multiple calls to fit method
            global_metrics_dict = Learner.__init_metrics(metrics)

            training_progress_bar = tqdm(total=len(train_loader), desc="{:12s}".format('Training'),
                                         dynamic_ncols=True, leave=True)
            # count number of steps
            step = 0

        # Nullify the parameter gradients
        optimizer.zero_grad(set_to_none=True)

        for batch_index, (x, y) in enumerate(train_loader):

            is_accumulating = batch_index % gradient_accumulation_steps != 0

            # If we are accumulating gradients, we do not need to step the optimizer
            with fabric.no_backward_sync(model, enabled=is_accumulating):
                outputs, x, y = self.__predictor.train_step(x, y, model=model,
                                                            device=fabric.device,
                                                            non_blocking=non_blocking)

                if isinstance(outputs, torch.Tensor) and outputs.ndim == 2 and outputs.shape[1] == 1:
                    y = y.view_as(outputs)

                loss = criterion(outputs, y)
                fabric.backward(loss)

            # If we are not accumulating gradients, we step the optimizer
            if not is_accumulating:

                # Gradient clipping
                if gradient_clip_value is not None:
                    fabric.clip_gradients(model, optimizer, clip_val=gradient_clip_value)
                elif gradient_clip_max_norm is not None:
                    fabric.clip_gradients(model, optimizer, max_norm=gradient_clip_max_norm)

                optimizer.step()

                if step_lr_scheduler is not None:
                    step_lr_scheduler.step()

                # Nullify the parameter gradients
                optimizer.zero_grad(set_to_none=True)

                local_batch_metrics_dict["loss"] = loss
                Learner.__update_metrics(outputs, y, metrics, local_batch_metrics_dict)

                # all_gather is used to aggregate the value across processes
                all_batch_metrics = fabric.all_gather(local_batch_metrics_dict)

                # Aggregate metrics across all processes
                if fabric.is_global_zero:
                    training_progress_bar.update(1)
                    step = step + 1
                    Learner.__update_metrics_with_simple_moving_average(all_batch_metrics, global_metrics_dict, step)
                    training_progress_bar.set_postfix({name: f'{round(value, 4)}' for name, value in  global_metrics_dict.items()})

        if fabric.is_global_zero:
            training_progress_bar.close()

        return global_metrics_dict


    def __validate(self, fabric: Fabric, model: torch.nn.Module, loader: torch.utils.data.DataLoader,
                   criterion: torch.nn.Module, metrics: Dict[str, torch.nn.Module] = None, non_blocking: bool= True):

        model.eval()
        local_batch_metrics_dict = Learner.__init_metrics(metrics)
        validation_progress_bar = None
        global_metrics_dict = OrderedDict()
        step = 0

        if fabric.is_global_zero:
            global_metrics_dict = Learner.__init_metrics(metrics)
            validation_progress_bar = tqdm(total=len(loader), desc="{:12s}".format('Validation'), dynamic_ncols=True, leave=True)

        with torch.no_grad():

            for batch_index, (x, y) in enumerate(loader):

                outputs, x, y = self.__predictor.eval_step(x, y, model=model,
                                                           device=fabric.device,
                                                           non_blocking=non_blocking)

                if isinstance(y, torch.Tensor):
                    y = y.to(fabric.device)

                if isinstance(outputs, torch.Tensor) and outputs.ndim == 2 and outputs.shape[1] == 1:
                    y = y.view_as(outputs)

                loss = criterion(outputs, y)

                local_batch_metrics_dict["loss"] = loss
                Learner.__update_metrics(outputs, y, metrics, local_batch_metrics_dict)

                # all_gather is used to aggregate the value across processes
                all_batch_metrics = fabric.all_gather(local_batch_metrics_dict)

                # Aggregate metrics across all processes
                if fabric.is_global_zero:
                    validation_progress_bar.update(1)
                    step = step + 1
                    Learner.__update_metrics_with_simple_moving_average(all_batch_metrics, global_metrics_dict, step)
                    validation_progress_bar.set_postfix({name: f'{round(value, 4)}' for name, value in  global_metrics_dict.items()})

        if fabric.is_global_zero:
            validation_progress_bar.close()

        return global_metrics_dict

    def predict(self, loader):
        predictions, targets = self.__predictor.predict(loader)
        return predictions, targets

    def predict_class(self, loader):
        predicted_class, probability, targets = self.__predictor.predict_class(loader)
        return predicted_class, probability, targets

    def extract_features(self, loader, no_of_features, features_csv_file, iterations=1,
                         target_known=True):

        fp = open(features_csv_file, 'w')
        csv_writer = csv.writer(fp)

        # define feature columns
        cols = ["feat_{}".format(i) for i in range(0, no_of_features)]

        if target_known:
            cols = ["class"] + cols

        csv_writer.writerow(cols)
        fp.flush()

        self.__model.eval()
        with torch.no_grad():
            for iteration in range(iterations):
                print('Iteration:', iteration + 1)
                for x, y in tqdm(loader, total=len(loader), desc='Feature Extraction'):

                    feature_set, x, y = self.__predictor.eval_step(x, y).cpu().numpy()

                    if target_known:
                        y = y.numpy().reshape(-1, 1)
                        feature_set = np.hstack([y, feature_set])

                    csv_writer.writerows(feature_set)
                    fp.flush()
        fp.close()

    def show_predictions(self, loader, image_inverse_transform=None, samples=9, cols=3, figsize=(10, 10),
                         target_known=True):

        self.__predictor.show_predictions(loader, image_inverse_transform=image_inverse_transform,
                                          samples=samples, cols=cols, figsize=figsize, target_known=target_known)
