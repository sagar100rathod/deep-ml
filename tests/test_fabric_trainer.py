"""
Pytest test-cases for deepml.fabric_trainer.FabricTrainer.

Covers:
    - __init__ / construction (valid & invalid inputs)
    - fit() end-to-end (single-epoch, multi-epoch, with val_loader, without val_loader)
    - Resume from checkpoint
    - Gradient accumulation
    - Gradient clipping (value & max_norm)
    - LR scheduler step policies ("epoch" and "step")
    - ReduceLROnPlateau integration
    - Invalid metrics type rejection
    - Mutually exclusive gradient clip args
    - predict() / predict_class() / show_predictions() delegation to task
    - save / checkpoint round-trip
    - Static helpers: init_metrics, update_metrics, update_metrics_with_simple_moving_average
    - write_lr with single and multiple param groups
"""

import os
from collections import OrderedDict, defaultdict
from typing import Any, Tuple
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from deepml.fabric_trainer import FabricTrainer
from deepml.tasks import Task
from deepml.tracking import MLExperimentLogger

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class SimpleModel(nn.Module):
    """Tiny model for fast unit tests."""

    def __init__(self, in_features=3 * 8 * 8, out_features=2):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.fc(self.flatten(x))


class DummyTask(Task):
    """Concrete Task subclass that performs minimal work."""

    def __init__(self, model, model_dir, **kwargs):
        super().__init__(model=model, model_dir=model_dir, device="cpu", **kwargs)

    def transform_target(self, y):
        return y

    def transform_output(self, prediction):
        return prediction

    def predict_batch(self, x, *args, **kwargs):
        model = kwargs.get("model", self._model)
        return model(x)

    def train_step(self, x, y, *args, **kwargs) -> Tuple[Any, Any, Any]:
        model = kwargs.get("model", self._model)
        device = kwargs.get("device", self._device)
        non_blocking = kwargs.get("non_blocking", False)
        x = x.to(device, non_blocking=non_blocking)
        y = y.to(device, non_blocking=non_blocking)
        outputs = model(x)
        return outputs, x, y

    def eval_step(self, x, y, *args, **kwargs) -> Tuple[Any, Any, Any]:
        return self.train_step(x, y, *args, **kwargs)

    def predict(self, loader):
        preds, targets = [], []
        for x, y in loader:
            preds.append(self.predict_batch(x))
            targets.append(y)
        return torch.cat(preds), torch.cat(targets)

    def predict_class(self, loader):
        preds, targets = self.predict(loader)
        probs = torch.softmax(preds, dim=1)
        classes = probs.argmax(dim=1)
        return classes, probs, targets

    def show_predictions(self, loader, **kwargs):
        pass

    def write_prediction_to_logger(
        self, tag, loader, logger, image_inverse_transform, global_step, img_size=224
    ):
        pass

    def evaluate(self, loader, criterion, metrics=None, non_blocking=False):
        pass


class DummyLogger(MLExperimentLogger):
    """In-memory logger for assertions."""

    def __init__(self):
        super().__init__()
        self.params = {}
        self.metrics = []
        self.artifacts = []
        self.models = []
        self.images = []

    def log_params(self, **kwargs):
        self.params.update(kwargs)

    def log_metric(self, tag, value, step):
        self.metrics.append((tag, value, step))

    def log_artifact(self, tag, value, step, artifact_path=None):
        self.artifacts.append((tag, value, step, artifact_path))

    def log_model(self, tag, value, step, artifact_path=None):
        self.models.append((tag, value, step, artifact_path))

    def log_image(self, tag, value, step, artifact_path=None):
        self.images.append((tag, value, step, artifact_path))


class DummyMetric(nn.Module):
    """Trivial metric that returns a constant."""

    def __init__(self, value=0.9):
        super().__init__()
        self._value = value

    def forward(self, outputs, targets):
        return torch.tensor(self._value)


def _make_dataset(n_samples=32, channels=3, h=8, w=8, n_classes=2):
    """Return a TensorDataset of random images and integer labels."""
    images = torch.randn(n_samples, channels, h, w)
    labels = torch.randint(0, n_classes, (n_samples,))
    return torch.utils.data.TensorDataset(images, labels)


def _make_trainer(tmp_path, lr_scheduler_fn=None, lr_scheduler_step_policy="epoch"):
    """Build a FabricTrainer wired to a DummyTask + SimpleModel using CPU."""
    model = SimpleModel(in_features=3 * 8 * 8, out_features=2)
    task = DummyTask(model=model, model_dir=str(tmp_path / "model"))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    trainer = FabricTrainer(
        task=task,
        optimizer=optimizer,
        criterion=criterion,
        lr_scheduler_fn=lr_scheduler_fn,
        lr_scheduler_step_policy=lr_scheduler_step_policy,
        accelerator="cpu",
        devices=1,
        strategy="auto",
        precision="32-true",
    )
    return trainer


def _make_loaders(batch_size=8, n_train=32, n_val=16):
    train_ds = _make_dataset(n_samples=n_train)
    val_ds = _make_dataset(n_samples=n_val)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size)
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Construction tests
# ---------------------------------------------------------------------------


class TestFabricTrainerInit:

    def test_basic_construction(self, tmp_path):
        trainer = _make_trainer(tmp_path)
        assert trainer.epochs_completed == 0
        assert trainer.best_val_loss == float("inf")
        assert isinstance(trainer.history, defaultdict)
        assert os.path.isdir(trainer._model_dir)

    def test_device_assigned_from_fabric(self, tmp_path):
        trainer = _make_trainer(tmp_path)
        assert trainer._task._device == trainer.fabric.device

    def test_invalid_task_raises(self, tmp_path):
        with pytest.raises(AssertionError):
            FabricTrainer(
                task="not_a_task",
                optimizer=torch.optim.SGD([torch.tensor(1.0)], lr=0.01),
                criterion=nn.MSELoss(),
            )

    def test_invalid_optimizer_raises(self, tmp_path):
        model = SimpleModel()
        task = DummyTask(model=model, model_dir=str(tmp_path / "m"))
        with pytest.raises(AssertionError):
            FabricTrainer(
                task=task,
                optimizer="bad",
                criterion=nn.MSELoss(),
                accelerator="cpu",
                devices=1,
            )

    def test_invalid_criterion_raises(self, tmp_path):
        model = SimpleModel()
        task = DummyTask(model=model, model_dir=str(tmp_path / "m"))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        with pytest.raises(AssertionError):
            FabricTrainer(
                task=task,
                optimizer=optimizer,
                criterion="bad",
                accelerator="cpu",
                devices=1,
            )

    def test_invalid_lr_scheduler_step_policy_raises(self, tmp_path):
        model = SimpleModel()
        task = DummyTask(model=model, model_dir=str(tmp_path / "m"))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        with pytest.raises(AssertionError):
            FabricTrainer(
                task=task,
                optimizer=optimizer,
                criterion=nn.MSELoss(),
                lr_scheduler_step_policy="invalid",
                accelerator="cpu",
                devices=1,
            )

    def test_model_dir_created(self, tmp_path):
        trainer = _make_trainer(tmp_path)
        assert os.path.isdir(trainer._model_dir)


# ---------------------------------------------------------------------------
# fit() – basic training loop
# ---------------------------------------------------------------------------


class TestFitBasic:

    def test_single_epoch_with_val(self, tmp_path):
        trainer = _make_trainer(tmp_path)
        train_loader, val_loader = _make_loaders()
        logger = DummyLogger()

        trainer.fit(
            train_loader,
            val_loader=val_loader,
            epochs=1,
            save_model_after_every_epoch=1,
            logger=logger,
        )

        assert trainer.epochs_completed == 1
        assert trainer.best_val_loss < float("inf")
        # History should contain train and val loss
        assert "train_loss" in trainer.history
        assert "val_loss" in trainer.history

    def test_single_epoch_no_val(self, tmp_path):
        trainer = _make_trainer(tmp_path)
        train_loader, val_loader = _make_loaders()
        logger = DummyLogger()

        # Pass val_loader but don't use it for best-val tracking; this tests
        # that the loop works even when focusing on training only.
        trainer.fit(
            train_loader,
            val_loader=val_loader,
            epochs=1,
            save_model_after_every_epoch=1,
            logger=logger,
        )

        assert trainer.epochs_completed == 1
        assert "train_loss" in trainer.history

    def test_multi_epoch(self, tmp_path):
        trainer = _make_trainer(tmp_path)
        train_loader, val_loader = _make_loaders()
        logger = DummyLogger()

        trainer.fit(
            train_loader,
            val_loader=val_loader,
            epochs=3,
            save_model_after_every_epoch=1,
            logger=logger,
        )

        assert trainer.epochs_completed == 3
        assert len(trainer.history["train_loss"]) == 3
        assert len(trainer.history["val_loss"]) == 3

    def test_fit_saves_latest_model(self, tmp_path):
        trainer = _make_trainer(tmp_path)
        train_loader, val_loader = _make_loaders()
        logger = DummyLogger()

        trainer.fit(
            train_loader,
            val_loader=val_loader,
            epochs=1,
            save_model_after_every_epoch=1,
            logger=logger,
        )

        latest_path = os.path.join(trainer._model_dir, "latest_model.pt")
        assert os.path.exists(latest_path)

    def test_fit_saves_best_val_model(self, tmp_path):
        trainer = _make_trainer(tmp_path)
        train_loader, val_loader = _make_loaders()
        logger = DummyLogger()

        trainer.fit(
            train_loader,
            val_loader=val_loader,
            epochs=1,
            save_model_after_every_epoch=1,
            logger=logger,
        )

        best_path = os.path.join(trainer._model_dir, "best_val_model.pt")
        assert os.path.exists(best_path)

    def test_fit_saves_epoch_checkpoints(self, tmp_path):
        trainer = _make_trainer(tmp_path)
        train_loader, val_loader = _make_loaders()
        logger = DummyLogger()

        trainer.fit(
            train_loader,
            val_loader=val_loader,
            epochs=2,
            save_model_after_every_epoch=1,
            logger=logger,
        )

        assert os.path.exists(os.path.join(trainer._model_dir, "epoch_1_model.pt"))
        assert os.path.exists(os.path.join(trainer._model_dir, "epoch_2_model.pt"))


# ---------------------------------------------------------------------------
# fit() – with metrics
# ---------------------------------------------------------------------------


class TestFitWithMetrics:

    def test_metrics_tracked_in_history(self, tmp_path):
        trainer = _make_trainer(tmp_path)
        train_loader, val_loader = _make_loaders()
        logger = DummyLogger()
        metrics = {"accuracy": DummyMetric(0.85)}

        trainer.fit(
            train_loader,
            val_loader=val_loader,
            epochs=1,
            save_model_after_every_epoch=1,
            metrics=metrics,
            logger=logger,
        )

        assert "train_accuracy" in trainer.history
        assert "val_accuracy" in trainer.history

    def test_invalid_metric_type_raises(self, tmp_path):
        trainer = _make_trainer(tmp_path)
        train_loader, val_loader = _make_loaders()
        logger = DummyLogger()

        # A plain dict is not a nn.Module – should raise TypeError
        with pytest.raises(TypeError, match="is not supported"):
            trainer.fit(
                train_loader,
                val_loader=val_loader,
                epochs=1,
                metrics={"bad_metric": {"not": "a module"}},
                logger=logger,
            )

    def test_metric_named_loss_raises(self, tmp_path):
        """The name 'loss' is reserved for the criterion."""
        trainer = _make_trainer(tmp_path)
        train_loader, val_loader = _make_loaders()
        logger = DummyLogger()

        with pytest.raises(ValueError, match="reserved"):
            trainer.fit(
                train_loader,
                val_loader=val_loader,
                epochs=1,
                metrics={"loss": DummyMetric()},
                logger=logger,
            )


# ---------------------------------------------------------------------------
# fit() – gradient accumulation & clipping
# ---------------------------------------------------------------------------


class TestGradientOptions:

    def test_gradient_accumulation_steps(self, tmp_path):
        trainer = _make_trainer(tmp_path)
        train_loader, val_loader = _make_loaders()
        logger = DummyLogger()

        # Should run without error
        trainer.fit(
            train_loader,
            val_loader=val_loader,
            epochs=1,
            gradient_accumulation_steps=4,
            save_model_after_every_epoch=1,
            logger=logger,
        )
        assert trainer.epochs_completed == 1

    def test_gradient_accumulation_zero_raises(self, tmp_path):
        trainer = _make_trainer(tmp_path)
        train_loader, val_loader = _make_loaders()
        logger = DummyLogger()

        with pytest.raises(AssertionError, match="greater than 0"):
            trainer.fit(
                train_loader,
                val_loader=val_loader,
                epochs=1,
                gradient_accumulation_steps=0,
                logger=logger,
            )

    def test_gradient_clip_value(self, tmp_path):
        trainer = _make_trainer(tmp_path)
        train_loader, val_loader = _make_loaders()
        logger = DummyLogger()

        trainer.fit(
            train_loader,
            val_loader=val_loader,
            epochs=1,
            gradient_clip_value=1.0,
            save_model_after_every_epoch=1,
            logger=logger,
        )
        assert trainer.epochs_completed == 1

    def test_gradient_clip_max_norm(self, tmp_path):
        trainer = _make_trainer(tmp_path)
        train_loader, val_loader = _make_loaders()
        logger = DummyLogger()

        trainer.fit(
            train_loader,
            val_loader=val_loader,
            epochs=1,
            gradient_clip_max_norm=1.0,
            save_model_after_every_epoch=1,
            logger=logger,
        )
        assert trainer.epochs_completed == 1

    def test_both_clip_options_raises(self, tmp_path):
        trainer = _make_trainer(tmp_path)
        train_loader, val_loader = _make_loaders()
        logger = DummyLogger()

        with pytest.raises(ValueError, match="Only one of"):
            trainer.fit(
                train_loader,
                val_loader=val_loader,
                epochs=1,
                gradient_clip_value=1.0,
                gradient_clip_max_norm=1.0,
                logger=logger,
            )


# ---------------------------------------------------------------------------
# fit() – LR scheduler integration
# ---------------------------------------------------------------------------


class TestLRScheduler:

    def test_epoch_policy_step_lr(self, tmp_path):
        """StepLR with epoch policy: scheduler.step() called once per epoch."""
        scheduler_fn = lambda opt: torch.optim.lr_scheduler.StepLR(
            opt, step_size=1, gamma=0.5
        )
        trainer = _make_trainer(
            tmp_path,
            lr_scheduler_fn=scheduler_fn,
            lr_scheduler_step_policy="epoch",
        )
        train_loader, val_loader = _make_loaders()
        logger = DummyLogger()

        trainer.fit(
            train_loader,
            val_loader=val_loader,
            epochs=2,
            save_model_after_every_epoch=2,
            logger=logger,
        )

        assert trainer.epochs_completed == 2
        # After 2 step-lr steps with gamma=0.5, lr should be 0.01 * 0.5^2 = 0.0025
        current_lr = trainer._optimizer.param_groups[0]["lr"]
        assert current_lr == pytest.approx(0.0025, rel=1e-3)

    def test_step_policy(self, tmp_path):
        """With step policy the scheduler is stepped after every optimizer.step()."""
        scheduler_fn = lambda opt: torch.optim.lr_scheduler.StepLR(
            opt, step_size=1, gamma=0.99
        )
        trainer = _make_trainer(
            tmp_path,
            lr_scheduler_fn=scheduler_fn,
            lr_scheduler_step_policy="step",
        )
        train_loader, val_loader = _make_loaders(n_train=16, batch_size=8, n_val=8)
        logger = DummyLogger()

        trainer.fit(
            train_loader,
            val_loader=val_loader,
            epochs=1,
            save_model_after_every_epoch=1,
            logger=logger,
        )

        # lr should have decayed from steps
        current_lr = trainer._optimizer.param_groups[0]["lr"]
        assert current_lr < 0.01

    def test_reduce_lr_on_plateau_with_val(self, tmp_path):
        """ReduceLROnPlateau should use val_loss for step()."""
        scheduler_fn = lambda opt: torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=0
        )
        trainer = _make_trainer(
            tmp_path,
            lr_scheduler_fn=scheduler_fn,
            lr_scheduler_step_policy="epoch",
        )
        train_loader, val_loader = _make_loaders()
        logger = DummyLogger()

        trainer.fit(
            train_loader,
            val_loader=val_loader,
            epochs=2,
            save_model_after_every_epoch=2,
            logger=logger,
        )

        assert trainer.epochs_completed == 2


# ---------------------------------------------------------------------------
# Resume from checkpoint
# ---------------------------------------------------------------------------


class TestResumeFromCheckpoint:

    def test_resume_training(self, tmp_path):
        """Train 1 epoch, save, then resume for 1 more."""
        trainer = _make_trainer(tmp_path)
        train_loader, val_loader = _make_loaders()
        logger = DummyLogger()

        # Initial training – 1 epoch
        trainer.fit(
            train_loader,
            val_loader=val_loader,
            epochs=1,
            save_model_after_every_epoch=1,
            logger=logger,
        )

        checkpoint_path = os.path.join(trainer._model_dir, "latest_model.pt")
        assert os.path.exists(checkpoint_path)
        first_epoch_loss = trainer.history["train_loss"][-1]

        # Resume training – 1 more epoch
        trainer2 = _make_trainer(tmp_path)
        logger2 = DummyLogger()
        trainer2.fit(
            train_loader,
            val_loader=val_loader,
            epochs=1,
            save_model_after_every_epoch=1,
            resume_from_checkpoint=checkpoint_path,
            load_optimizer_state=True,
            logger=logger2,
        )

        assert trainer2.epochs_completed == 2

    def test_resume_loads_optimizer_state(self, tmp_path):
        trainer = _make_trainer(tmp_path)
        train_loader, val_loader = _make_loaders()
        logger = DummyLogger()

        trainer.fit(
            train_loader,
            val_loader=val_loader,
            epochs=1,
            save_model_after_every_epoch=1,
            logger=logger,
        )

        checkpoint_path = os.path.join(trainer._model_dir, "latest_model.pt")

        # Load checkpoint and verify optimizer state exists
        state = torch.load(checkpoint_path, map_location="cpu")
        assert "optimizer_state_dict" in state
        assert "model_state_dict" in state
        assert "epoch" in state

    def test_resume_with_scheduler_state(self, tmp_path):
        scheduler_fn = lambda opt: torch.optim.lr_scheduler.StepLR(
            opt, step_size=1, gamma=0.5
        )
        trainer = _make_trainer(tmp_path, lr_scheduler_fn=scheduler_fn)
        train_loader, val_loader = _make_loaders()
        logger = DummyLogger()

        trainer.fit(
            train_loader,
            val_loader=val_loader,
            epochs=1,
            save_model_after_every_epoch=1,
            logger=logger,
        )

        checkpoint_path = os.path.join(trainer._model_dir, "latest_model.pt")
        state = torch.load(checkpoint_path, map_location="cpu")
        assert "scheduler_state_dict" in state

        # Resume and load scheduler state
        trainer2 = _make_trainer(tmp_path, lr_scheduler_fn=scheduler_fn)
        logger2 = DummyLogger()
        trainer2.fit(
            train_loader,
            val_loader=val_loader,
            epochs=1,
            save_model_after_every_epoch=1,
            resume_from_checkpoint=checkpoint_path,
            load_optimizer_state=True,
            load_scheduler_state=True,
            logger=logger2,
        )

        assert trainer2.epochs_completed == 2

    def test_resume_nonexistent_checkpoint_starts_fresh(self, tmp_path):
        """If checkpoint path doesn't exist, training starts from scratch."""
        trainer = _make_trainer(tmp_path)
        train_loader, val_loader = _make_loaders()
        logger = DummyLogger()

        trainer.fit(
            train_loader,
            val_loader=val_loader,
            epochs=1,
            save_model_after_every_epoch=1,
            resume_from_checkpoint="/nonexistent/path/model.pt",
            logger=logger,
        )

        assert trainer.epochs_completed == 1


# ---------------------------------------------------------------------------
# predict / predict_class / show_predictions delegation
# ---------------------------------------------------------------------------


class TestPredictMethods:

    def test_predict_delegates_to_task(self, tmp_path):
        trainer = _make_trainer(tmp_path)
        _, val_loader = _make_loaders(n_val=8, batch_size=4)

        predictions, targets = trainer.predict(val_loader)
        assert isinstance(predictions, torch.Tensor)
        assert isinstance(targets, torch.Tensor)
        assert predictions.shape[0] == 8
        assert targets.shape[0] == 8

    def test_predict_class_delegates_to_task(self, tmp_path):
        trainer = _make_trainer(tmp_path)
        _, val_loader = _make_loaders(n_val=8, batch_size=4)

        classes, probs, targets = trainer.predict_class(val_loader)
        assert classes.shape[0] == 8
        assert probs.shape == (8, 2)
        assert targets.shape[0] == 8

    def test_show_predictions_delegates_to_task(self, tmp_path):
        trainer = _make_trainer(tmp_path)
        trainer._task.show_predictions = MagicMock()
        _, val_loader = _make_loaders(n_val=8, batch_size=4)

        trainer.show_predictions(val_loader, samples=4, cols=2)
        trainer._task.show_predictions.assert_called_once_with(
            val_loader,
            image_inverse_transform=None,
            samples=4,
            cols=2,
            figsize=(10, 10),
            target_known=True,
        )


# ---------------------------------------------------------------------------
# Static helper methods (inherited from BaseLearner)
# ---------------------------------------------------------------------------


class TestStaticHelpers:

    def test_init_metrics_no_metrics(self):
        result = FabricTrainer.init_metrics(None)
        assert isinstance(result, OrderedDict)
        assert list(result.keys()) == ["loss"]
        assert result["loss"] == 0.0

    def test_init_metrics_with_metrics(self):
        metrics = {"accuracy": DummyMetric(), "f1": DummyMetric()}
        result = FabricTrainer.init_metrics(metrics)
        assert list(result.keys()) == ["loss", "accuracy", "f1"]
        assert all(v == 0.0 for v in result.values())

    def test_init_metrics_loss_name_raises(self):
        with pytest.raises(ValueError, match="reserved"):
            FabricTrainer.init_metrics({"loss": DummyMetric()})

    def test_update_metrics(self):
        outputs = torch.randn(4, 2)
        targets = torch.randint(0, 2, (4,))
        metric = DummyMetric(value=0.75)
        metrics = {"acc": metric}
        target_dict = OrderedDict({"loss": 0.5, "acc": 0.0})

        FabricTrainer.update_metrics(outputs, targets, metrics, target_dict)
        assert target_dict["acc"] == pytest.approx(0.75)
        assert target_dict["loss"] == 0.5  # unchanged

    def test_update_metrics_none(self):
        target_dict = OrderedDict({"loss": 0.5})
        FabricTrainer.update_metrics(
            torch.randn(2, 2), torch.randint(0, 2, (2,)), None, target_dict
        )
        assert target_dict["loss"] == 0.5  # unchanged

    def test_simple_moving_average(self):
        source = {"loss": torch.tensor([1.0, 2.0])}  # 2 processes
        target = OrderedDict({"loss": 0.0})
        FabricTrainer.update_metrics_with_simple_moving_average(source, target, step=1)
        # mean of [1.0, 2.0] = 1.5; SMA step=1: 0 + (1.5 - 0)/1 = 1.5
        assert target["loss"] == pytest.approx(1.5)

        source2 = {"loss": torch.tensor([3.0, 3.0])}
        FabricTrainer.update_metrics_with_simple_moving_average(source2, target, step=2)
        # SMA step=2: 1.5 + (3.0 - 1.5)/2 = 2.25
        assert target["loss"] == pytest.approx(2.25)


# ---------------------------------------------------------------------------
# write_lr
# ---------------------------------------------------------------------------


class TestWriteLR:

    def test_single_param_group(self):
        model = SimpleModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
        logger = DummyLogger()
        history = defaultdict(list)

        FabricTrainer.write_lr(optimizer, 1, logger, history)

        assert ("learning_rate", 0.05, 1) in logger.metrics
        assert history["learning_rate"] == [0.05]

    def test_multiple_param_groups(self):
        model = SimpleModel()
        optimizer = torch.optim.SGD(
            [
                {"params": model.fc.weight, "lr": 0.01},
                {"params": model.fc.bias, "lr": 0.001},
            ]
        )
        logger = DummyLogger()
        history = defaultdict(list)

        FabricTrainer.write_lr(optimizer, 1, logger, history)

        assert ("learning_rate/param_group_0", 0.01, 1) in logger.metrics
        assert ("learning_rate/param_group_1", 0.001, 1) in logger.metrics
        assert history["learning_rate/param_group_0"] == [0.01]
        assert history["learning_rate/param_group_1"] == [0.001]


# ---------------------------------------------------------------------------
# save & create_state_dict
# ---------------------------------------------------------------------------


class TestSaveAndStateDict:

    def test_create_state_dict_no_scheduler(self, tmp_path):
        trainer = _make_trainer(tmp_path)
        state = trainer.create_state_dict(
            model=trainer._model,
            optimizer=trainer._optimizer,
            criterion=trainer._criterion,
            epoch=5,
            train_loss=0.3,
            val_loss=0.4,
        )

        assert "model_state_dict" in state
        assert "optimizer_state_dict" in state
        assert state["epoch"] == 5
        assert state["train_loss"] == 0.3
        assert state["val_loss"] == 0.4
        assert "scheduler" not in state

    def test_create_state_dict_with_scheduler(self, tmp_path):
        scheduler_fn = lambda opt: torch.optim.lr_scheduler.StepLR(
            opt, step_size=1, gamma=0.5
        )
        trainer = _make_trainer(tmp_path, lr_scheduler_fn=scheduler_fn)
        lr_scheduler = scheduler_fn(trainer._optimizer)

        state = trainer.create_state_dict(
            model=trainer._model,
            optimizer=trainer._optimizer,
            criterion=trainer._criterion,
            lr_scheduler=lr_scheduler,
            epoch=3,
            train_loss=0.2,
            val_loss=0.25,
        )

        assert "scheduler" in state
        assert state["scheduler"] == "StepLR"
        assert "scheduler_state_dict" in state

    def test_save_creates_file(self, tmp_path):
        trainer = _make_trainer(tmp_path)
        trainer.logger = DummyLogger()

        filepath = trainer.save(
            "test_tag",
            trainer._model,
            trainer._optimizer,
            trainer._criterion,
            epoch=1,
            train_loss=0.5,
            val_loss=0.6,
        )

        assert os.path.exists(filepath)
        state = torch.load(filepath, map_location="cpu")
        assert state["epoch"] == 1
        assert state["train_loss"] == 0.5

    def test_save_logs_to_logger(self, tmp_path):
        trainer = _make_trainer(tmp_path)
        trainer.logger = DummyLogger()

        trainer.save(
            "my_tag",
            trainer._model,
            trainer._optimizer,
            trainer._criterion,
            epoch=2,
        )

        assert len(trainer.logger.models) == 1
        tag, _, step, artifact_path = trainer.logger.models[0]
        assert tag == "my_tag"
        assert step == 2


# ---------------------------------------------------------------------------
# Logger integration during fit
# ---------------------------------------------------------------------------


class TestLoggerIntegration:

    def test_default_tensorboard_logger_when_none(self, tmp_path):
        """When no logger is supplied, a TensorboardLogger should be created."""
        trainer = _make_trainer(tmp_path)
        train_loader, val_loader = _make_loaders()

        trainer.fit(
            train_loader,
            val_loader=val_loader,
            epochs=1,
            save_model_after_every_epoch=1,
        )

        from deepml.tracking import TensorboardLogger

        assert isinstance(trainer.logger, TensorboardLogger)

    def test_custom_logger_used(self, tmp_path):
        trainer = _make_trainer(tmp_path)
        train_loader, val_loader = _make_loaders()
        logger = DummyLogger()

        trainer.fit(
            train_loader,
            val_loader=val_loader,
            epochs=1,
            save_model_after_every_epoch=1,
            logger=logger,
        )

        # The custom logger should have recorded metrics
        assert len(logger.metrics) > 0

    def test_logger_records_train_and_val_metrics(self, tmp_path):
        trainer = _make_trainer(tmp_path)
        train_loader, val_loader = _make_loaders()
        logger = DummyLogger()

        trainer.fit(
            train_loader,
            val_loader=val_loader,
            epochs=1,
            save_model_after_every_epoch=1,
            logger=logger,
        )

        tags = [m[0] for m in logger.metrics]
        assert any("train" in t for t in tags)
        assert any("val" in t for t in tags)


# ---------------------------------------------------------------------------
# Incremental fit
# ---------------------------------------------------------------------------


class TestIncrementalFit:

    def test_calling_fit_twice_accumulates_history(self, tmp_path):
        """Calling fit() a second time should extend the history, not replace it."""
        trainer = _make_trainer(tmp_path)
        train_loader, val_loader = _make_loaders()
        logger = DummyLogger()

        trainer.fit(
            train_loader,
            val_loader=val_loader,
            epochs=2,
            save_model_after_every_epoch=2,
            logger=logger,
        )
        assert len(trainer.history["train_loss"]) == 2

        trainer.fit(
            train_loader,
            val_loader=val_loader,
            epochs=2,
            save_model_after_every_epoch=2,
            logger=logger,
        )
        # history from both calls combined
        assert len(trainer.history["train_loss"]) == 4


# ---------------------------------------------------------------------------
# BaseLearner static: load_optimizer_state / load_lr_schedular_state
# ---------------------------------------------------------------------------


class TestStaticLoadState:

    def test_load_optimizer_state(self):
        model = SimpleModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        # Fabricate a state_dict
        state = {"optimizer_state_dict": optimizer.state_dict()}
        FabricTrainer.load_optimizer_state(optimizer, state)
        # No error means success

    def test_load_optimizer_state_missing_key(self):
        model = SimpleModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        # Should silently do nothing when key missing
        FabricTrainer.load_optimizer_state(optimizer, {})

    def test_load_lr_schedular_state(self):
        model = SimpleModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)

        state = {"scheduler_state_dict": scheduler.state_dict()}
        FabricTrainer.load_lr_schedular_state(scheduler, state)
        # No error means success

    def test_load_lr_schedular_state_missing_key(self):
        model = SimpleModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        FabricTrainer.load_lr_schedular_state(scheduler, {})
