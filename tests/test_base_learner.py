"""
Pytest test-cases for deepml.base.BaseLearner.

BaseLearner is abstract, so we create a minimal concrete subclass
(ConcreteLearner) that only satisfies the ABC contract while letting us
test every non-abstract method directly:

    Construction & attribute wiring
    ├── __init__ happy-path
    ├── task type assertion
    ├── mutual-exclusion of lr_scheduler / lr_scheduler_fn
    ├── properties forwarded from task
    ├── logger defaults to None
    Setters
    ├── set_optimizer (valid / invalid)
    ├── set_criterion (valid / invalid)
    ├── set_lr_scheduler_policy (valid / invalid)
    Static helpers – load state
    ├── load_optimizer_state (present / missing key)
    ├── load_lr_schedular_state (present / missing key)
    create_state_dict
    ├── without lr_scheduler
    ├── with lr_scheduler
    ├── default arg values
    save
    ├── creates file on disk
    ├── state dict contents round-trip
    ├── delegates to logger.log_model
    init_metrics
    ├── None metrics
    ├── valid metrics
    ├── reserved "loss" name
    update_metrics
    ├── populates target dict
    ├── None metrics dict is no-op
    update_metrics_with_simple_moving_average
    ├── single step
    ├── multi step convergence
    write_metrics_to_logger
    ├── writes all metrics to logger and history
    write_lr
    ├── single param group
    ├── multiple param groups
    log_metrics
    ├── writes train + val metrics
    ├── writes prediction images when logger_img_size is set
    ├── skips images when logger_img_size is None
    fit / predict
    ├── raise NotImplementedError
"""

import os
from collections import OrderedDict, defaultdict
from typing import Any, Tuple
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from deepml.base import BaseLearner
from deepml.tasks import Task
from deepml.tracking import MLExperimentLogger

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _SimpleModel(nn.Module):
    def __init__(self, in_features=16, out_features=2):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.fc(x)


class _DummyTask(Task):
    """Minimal concrete Task so we can instantiate BaseLearner."""

    def __init__(self, model, model_dir, **kw):
        super().__init__(model=model, model_dir=model_dir, device="cpu", **kw)

    # -- required abstract methods --
    def transform_target(self, y):
        return y

    def transform_output(self, prediction):
        return prediction

    def predict_batch(self, x, *a, **kw):
        return self._model(x)

    def train_step(self, x, y, *a, **kw) -> Tuple[Any, Any, Any]:
        return self._model(x), x, y

    def eval_step(self, x, y, *a, **kw) -> Tuple[Any, Any, Any]:
        return self.train_step(x, y)

    def predict(self, loader):
        pass

    def predict_class(self, loader):
        pass

    def show_predictions(self, loader, **kw):
        pass

    def write_prediction_to_logger(self, *a, **kw):
        pass

    def evaluate(self, loader, criterion, metrics=None, non_blocking=False):
        pass


class _DummyLogger(MLExperimentLogger):
    """In-memory logger for assertions."""

    def __init__(self):
        super().__init__()
        self.metrics: list = []
        self.models: list = []

    def log_params(self, **kw):
        pass

    def log_metric(self, tag, value, step):
        self.metrics.append((tag, value, step))

    def log_artifact(self, tag, value, step, artifact_path=None):
        pass

    def log_model(self, tag, value, step, artifact_path=None):
        self.models.append((tag, value, step, artifact_path))

    def log_image(self, tag, value, step, artifact_path=None):
        pass


class ConcreteLearner(BaseLearner):
    """Thin concrete subclass that satisfies the ABC."""

    def fit(self, *args, **kwargs):
        raise NotImplementedError

    def predict(self, *args, **kwargs):
        raise NotImplementedError


class _ConstantMetric(nn.Module):
    """Returns a fixed scalar tensor."""

    def __init__(self, value: float = 0.9):
        super().__init__()
        self._v = value

    def forward(self, outputs, targets):
        return torch.tensor(self._v)


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def _make_learner(
    tmp_path,
    lr_scheduler=None,
    lr_scheduler_fn=None,
    lr_scheduler_step_policy="epoch",
):
    model = _SimpleModel()
    task = _DummyTask(model=model, model_dir=str(tmp_path / "model"))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    learner = ConcreteLearner(
        task=task,
        optimizer=optimizer,
        criterion=criterion,
        lr_scheduler=lr_scheduler,
        lr_scheduler_fn=lr_scheduler_fn,
        lr_scheduler_step_policy=lr_scheduler_step_policy,
    )
    return learner


# ===================================================================
# Construction & attribute wiring
# ===================================================================


class TestBaseLearnerInit:

    def test_happy_path(self, tmp_path):
        learner = _make_learner(tmp_path)
        assert learner._task is not None
        assert learner._model is learner._task.model
        assert learner._model_dir == learner._task.model_dir
        assert learner._model_file_name == learner._task.model_file_name
        assert learner._lr_scheduler is None
        assert learner._lr_scheduler_fn is None
        assert learner._lr_scheduler_step_policy == "epoch"
        assert learner.logger is None

    def test_task_type_assertion(self, tmp_path):
        with pytest.raises(AssertionError):
            ConcreteLearner(
                task="not_a_task",
                optimizer=torch.optim.SGD(
                    [torch.zeros(1, requires_grad=True)], lr=0.01
                ),
                criterion=nn.MSELoss(),
            )

    def test_lr_scheduler_and_fn_mutual_exclusion(self, tmp_path):
        model = _SimpleModel()
        task = _DummyTask(model=model, model_dir=str(tmp_path / "m"))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        scheduler_fn = lambda opt: torch.optim.lr_scheduler.StepLR(opt, step_size=1)

        with pytest.raises(AssertionError, match="Either"):
            ConcreteLearner(
                task=task,
                optimizer=optimizer,
                criterion=nn.MSELoss(),
                lr_scheduler=scheduler,
                lr_scheduler_fn=scheduler_fn,
            )

    def test_lr_scheduler_only(self, tmp_path):
        model = _SimpleModel()
        task = _DummyTask(model=model, model_dir=str(tmp_path / "m"))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)

        learner = ConcreteLearner(
            task=task,
            optimizer=optimizer,
            criterion=nn.MSELoss(),
            lr_scheduler=scheduler,
        )
        assert learner._lr_scheduler is scheduler
        assert learner._lr_scheduler_fn is None

    def test_lr_scheduler_fn_only(self, tmp_path):
        fn = lambda opt: torch.optim.lr_scheduler.StepLR(opt, step_size=1)
        learner = _make_learner(tmp_path, lr_scheduler_fn=fn)
        assert learner._lr_scheduler is None
        assert learner._lr_scheduler_fn is fn

    def test_properties_forwarded_from_task(self, tmp_path):
        learner = _make_learner(tmp_path)
        assert learner._model_dir == str(tmp_path / "model")
        assert learner._model_file_name == "latest_model.pt"


# ===================================================================
# Setters
# ===================================================================


class TestSetters:

    def test_set_optimizer_valid(self, tmp_path):
        learner = _make_learner(tmp_path)
        new_opt = torch.optim.Adam(learner._model.parameters(), lr=0.001)
        learner.set_optimizer(new_opt)
        assert learner._optimizer is new_opt

    def test_set_optimizer_invalid(self, tmp_path):
        learner = _make_learner(tmp_path)
        with pytest.raises(AssertionError):
            learner.set_optimizer("not_an_optimizer")

    def test_set_criterion_valid(self, tmp_path):
        learner = _make_learner(tmp_path)
        new_crit = nn.MSELoss()
        learner.set_criterion(new_crit)
        assert learner._criterion is new_crit

    def test_set_criterion_invalid(self, tmp_path):
        learner = _make_learner(tmp_path)
        with pytest.raises(AssertionError):
            learner.set_criterion("not_a_module")

    def test_set_lr_scheduler_policy_epoch(self, tmp_path):
        learner = _make_learner(tmp_path)
        learner.set_lr_scheduler_policy("epoch")
        assert learner._lr_scheduler_step_policy == "epoch"

    def test_set_lr_scheduler_policy_step(self, tmp_path):
        learner = _make_learner(tmp_path)
        learner.set_lr_scheduler_policy("step")
        assert learner._lr_scheduler_step_policy == "step"

    def test_set_lr_scheduler_policy_invalid_value(self, tmp_path):
        learner = _make_learner(tmp_path)
        with pytest.raises(AssertionError):
            learner.set_lr_scheduler_policy("batch")

    def test_set_lr_scheduler_policy_non_string(self, tmp_path):
        learner = _make_learner(tmp_path)
        with pytest.raises(AssertionError):
            learner.set_lr_scheduler_policy(123)


# ===================================================================
# Static helpers – load state
# ===================================================================


class TestLoadState:

    def test_load_optimizer_state_present(self):
        model = _SimpleModel()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        # Do a dummy step so optimizer has state
        x = torch.randn(2, 16)
        y = torch.randint(0, 2, (2,))
        loss = nn.CrossEntropyLoss()(model(x), y)
        loss.backward()
        opt.step()

        saved_state = {"optimizer_state_dict": opt.state_dict()}

        # Create fresh optimizer and load
        opt2 = torch.optim.SGD(model.parameters(), lr=0.05)
        BaseLearner.load_optimizer_state(opt2, saved_state)
        # LR from saved state should override
        assert opt2.state_dict()["param_groups"][0]["lr"] == 0.01

    def test_load_optimizer_state_missing_key(self):
        model = _SimpleModel()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        original_state = opt.state_dict()
        BaseLearner.load_optimizer_state(opt, {})
        # State should be untouched
        assert opt.state_dict() == original_state

    def test_load_lr_schedular_state_present(self):
        model = _SimpleModel()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.5)
        sched.step()  # last_epoch becomes 1
        saved = {"scheduler_state_dict": sched.state_dict()}

        opt2 = torch.optim.SGD(model.parameters(), lr=0.01)
        sched2 = torch.optim.lr_scheduler.StepLR(opt2, step_size=1, gamma=0.5)
        BaseLearner.load_lr_schedular_state(sched2, saved)
        assert sched2.state_dict()["last_epoch"] == sched.state_dict()["last_epoch"]

    def test_load_lr_schedular_state_missing_key(self):
        model = _SimpleModel()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1)
        original = sched.state_dict().copy()
        BaseLearner.load_lr_schedular_state(sched, {})
        assert sched.state_dict()["last_epoch"] == original["last_epoch"]


# ===================================================================
# create_state_dict
# ===================================================================


class TestCreateStateDict:

    def test_without_scheduler(self, tmp_path):
        learner = _make_learner(tmp_path)
        sd = learner.create_state_dict(
            model=learner._model,
            optimizer=learner._optimizer,
            criterion=learner._criterion,
            epoch=10,
            train_loss=0.25,
            val_loss=0.30,
        )
        assert "model_state_dict" in sd
        assert "optimizer_state_dict" in sd
        assert sd["optimizer"] == "SGD"
        assert sd["criterion"] == "CrossEntropyLoss"
        assert sd["epoch"] == 10
        assert sd["train_loss"] == 0.25
        assert sd["val_loss"] == 0.30
        assert "scheduler" not in sd
        assert "scheduler_state_dict" not in sd

    def test_with_scheduler(self, tmp_path):
        learner = _make_learner(tmp_path)
        sched = torch.optim.lr_scheduler.StepLR(
            learner._optimizer, step_size=1, gamma=0.5
        )
        sd = learner.create_state_dict(
            model=learner._model,
            optimizer=learner._optimizer,
            criterion=learner._criterion,
            lr_scheduler=sched,
            epoch=3,
        )
        assert sd["scheduler"] == "StepLR"
        assert "scheduler_state_dict" in sd

    def test_default_values(self, tmp_path):
        learner = _make_learner(tmp_path)
        sd = learner.create_state_dict(
            model=learner._model,
            optimizer=learner._optimizer,
            criterion=learner._criterion,
        )
        assert sd["epoch"] == -1
        assert sd["train_loss"] == float("inf")
        assert sd["val_loss"] == float("inf")

    def test_model_state_dict_matches(self, tmp_path):
        learner = _make_learner(tmp_path)
        sd = learner.create_state_dict(
            model=learner._model,
            optimizer=learner._optimizer,
            criterion=learner._criterion,
        )
        for key in learner._model.state_dict():
            assert torch.equal(
                sd["model_state_dict"][key], learner._model.state_dict()[key]
            )

    def test_optimizer_state_dict_matches(self, tmp_path):
        learner = _make_learner(tmp_path)
        sd = learner.create_state_dict(
            model=learner._model,
            optimizer=learner._optimizer,
            criterion=learner._criterion,
        )
        assert sd["optimizer_state_dict"] == learner._optimizer.state_dict()


# ===================================================================
# save
# ===================================================================


class TestSave:

    def test_creates_file_on_disk(self, tmp_path):
        learner = _make_learner(tmp_path)
        learner.logger = _DummyLogger()

        filepath = learner.save(
            "checkpoint",
            learner._model,
            learner._optimizer,
            learner._criterion,
            epoch=5,
            train_loss=0.1,
            val_loss=0.2,
        )

        assert filepath.endswith(".pt")
        assert os.path.exists(filepath)

    def test_saved_state_dict_round_trip(self, tmp_path):
        learner = _make_learner(tmp_path)
        learner.logger = _DummyLogger()

        filepath = learner.save(
            "ckpt",
            learner._model,
            learner._optimizer,
            learner._criterion,
            epoch=7,
            train_loss=0.4,
            val_loss=0.5,
        )

        loaded = torch.load(filepath, map_location="cpu")
        assert loaded["epoch"] == 7
        assert loaded["train_loss"] == pytest.approx(0.4)
        assert loaded["val_loss"] == pytest.approx(0.5)
        assert loaded["optimizer"] == "SGD"
        assert loaded["criterion"] == "CrossEntropyLoss"

        # model weights round-trip
        model2 = _SimpleModel()
        model2.load_state_dict(loaded["model_state_dict"])
        for p1, p2 in zip(learner._model.parameters(), model2.parameters()):
            assert torch.equal(p1, p2)

    def test_delegates_to_logger_log_model(self, tmp_path):
        learner = _make_learner(tmp_path)
        learner.logger = _DummyLogger()

        learner.save(
            "my_tag",
            learner._model,
            learner._optimizer,
            learner._criterion,
            epoch=2,
        )

        assert len(learner.logger.models) == 1
        tag, model, step, artifact_path = learner.logger.models[0]
        assert tag == "my_tag"
        assert model is learner._model
        assert step == 2
        assert artifact_path.endswith("my_tag.pt")

    def test_save_with_scheduler(self, tmp_path):
        learner = _make_learner(tmp_path)
        learner.logger = _DummyLogger()
        sched = torch.optim.lr_scheduler.StepLR(
            learner._optimizer, step_size=1, gamma=0.5
        )

        filepath = learner.save(
            "sched_ckpt",
            learner._model,
            learner._optimizer,
            learner._criterion,
            lr_scheduler=sched,
            epoch=1,
        )

        loaded = torch.load(filepath, map_location="cpu")
        assert "scheduler" in loaded
        assert "scheduler_state_dict" in loaded

    def test_save_default_losses(self, tmp_path):
        learner = _make_learner(tmp_path)
        learner.logger = _DummyLogger()

        filepath = learner.save(
            "default_losses",
            learner._model,
            learner._optimizer,
            learner._criterion,
        )

        loaded = torch.load(filepath, map_location="cpu")
        assert loaded["train_loss"] == float("inf")
        assert loaded["val_loss"] == float("inf")
        assert loaded["epoch"] == -1


# ===================================================================
# init_metrics
# ===================================================================


class TestInitMetrics:

    def test_none_metrics(self):
        result = BaseLearner.init_metrics(None)
        assert isinstance(result, OrderedDict)
        assert list(result.keys()) == ["loss"]
        assert result["loss"] == 0.0

    def test_valid_metrics(self):
        metrics = {"accuracy": _ConstantMetric(), "f1_score": _ConstantMetric()}
        result = BaseLearner.init_metrics(metrics)
        assert list(result.keys()) == ["loss", "accuracy", "f1_score"]
        assert all(v == 0.0 for v in result.values())

    def test_single_metric(self):
        result = BaseLearner.init_metrics({"recall": _ConstantMetric()})
        assert list(result.keys()) == ["loss", "recall"]

    def test_reserved_loss_name_raises(self):
        with pytest.raises(ValueError, match="reserved"):
            BaseLearner.init_metrics({"loss": _ConstantMetric()})

    def test_empty_dict(self):
        result = BaseLearner.init_metrics({})
        assert list(result.keys()) == ["loss"]

    def test_order_preserved(self):
        metrics = OrderedDict(
            [("z_metric", _ConstantMetric()), ("a_metric", _ConstantMetric())]
        )
        result = BaseLearner.init_metrics(metrics)
        assert list(result.keys()) == ["loss", "z_metric", "a_metric"]


# ===================================================================
# update_metrics
# ===================================================================


class TestUpdateMetrics:

    def test_populates_target_dict(self):
        outputs = torch.randn(4, 2)
        targets = torch.randint(0, 2, (4,))
        metrics = {"acc": _ConstantMetric(0.85), "f1": _ConstantMetric(0.70)}
        target_dict = OrderedDict({"loss": 0.5, "acc": 0.0, "f1": 0.0})

        BaseLearner.update_metrics(outputs, targets, metrics, target_dict)

        assert target_dict["acc"] == pytest.approx(0.85)
        assert target_dict["f1"] == pytest.approx(0.70)
        assert target_dict["loss"] == 0.5  # untouched

    def test_none_metrics_is_noop(self):
        target_dict = OrderedDict({"loss": 1.0})
        BaseLearner.update_metrics(torch.randn(2, 2), torch.zeros(2), None, target_dict)
        assert target_dict["loss"] == 1.0

    def test_empty_metrics_dict(self):
        target_dict = OrderedDict({"loss": 0.3})
        BaseLearner.update_metrics(torch.randn(2, 2), torch.zeros(2), {}, target_dict)
        assert target_dict["loss"] == 0.3

    def test_metric_receives_correct_args(self):
        outputs = torch.randn(4, 2)
        targets = torch.randint(0, 2, (4,))
        mock_metric = MagicMock(return_value=torch.tensor(0.99))
        metrics = {"m": mock_metric}
        target_dict = OrderedDict({"loss": 0.0, "m": 0.0})

        BaseLearner.update_metrics(outputs, targets, metrics, target_dict)

        mock_metric.assert_called_once()
        call_outputs, call_targets = mock_metric.call_args[0]
        assert torch.equal(call_outputs, outputs)
        assert torch.equal(call_targets, targets)


# ===================================================================
# update_metrics_with_simple_moving_average
# ===================================================================


class TestSMA:

    def test_first_step(self):
        source = {"loss": torch.tensor([2.0, 4.0])}  # mean = 3.0
        target = OrderedDict({"loss": 0.0})
        BaseLearner.update_metrics_with_simple_moving_average(source, target, step=1)
        # SMA: 0.0 + (3.0 - 0.0) / 1 = 3.0
        assert target["loss"] == pytest.approx(3.0)

    def test_multi_step_convergence(self):
        target = OrderedDict({"loss": 0.0})

        BaseLearner.update_metrics_with_simple_moving_average(
            {"loss": torch.tensor(4.0)}, target, step=1
        )
        assert target["loss"] == pytest.approx(4.0)

        BaseLearner.update_metrics_with_simple_moving_average(
            {"loss": torch.tensor(6.0)}, target, step=2
        )
        # 4.0 + (6.0 - 4.0) / 2 = 5.0
        assert target["loss"] == pytest.approx(5.0)

        BaseLearner.update_metrics_with_simple_moving_average(
            {"loss": torch.tensor(8.0)}, target, step=3
        )
        # 5.0 + (8.0 - 5.0) / 3 = 6.0
        assert target["loss"] == pytest.approx(6.0)

    def test_multiple_metrics(self):
        source = {"loss": torch.tensor(1.0), "acc": torch.tensor(0.8)}
        target = OrderedDict({"loss": 0.0, "acc": 0.0})
        BaseLearner.update_metrics_with_simple_moving_average(source, target, step=1)
        assert target["loss"] == pytest.approx(1.0)
        assert target["acc"] == pytest.approx(0.8)

    def test_tensor_with_multiple_elements_uses_mean(self):
        source = {"loss": torch.tensor([1.0, 3.0, 5.0])}  # mean = 3.0
        target = OrderedDict({"loss": 0.0})
        BaseLearner.update_metrics_with_simple_moving_average(source, target, step=1)
        assert target["loss"] == pytest.approx(3.0)


# ===================================================================
# write_metrics_to_logger
# ===================================================================


class TestWriteMetricsToLogger:

    def test_writes_all_metrics(self):
        logger = _DummyLogger()
        history = defaultdict(list)
        metrics = {"loss": 0.5, "accuracy": 0.9}

        BaseLearner.write_metrics_to_logger(metrics, "train", 1, logger, history)

        assert ("loss/train", 0.5, 1) in logger.metrics
        assert ("accuracy/train", 0.9, 1) in logger.metrics
        assert history["train_loss"] == [0.5]
        assert history["train_accuracy"] == [0.9]

    def test_val_tag(self):
        logger = _DummyLogger()
        history = defaultdict(list)
        metrics = {"loss": 0.3}

        BaseLearner.write_metrics_to_logger(metrics, "val", 5, logger, history)

        assert ("loss/val", 0.3, 5) in logger.metrics
        assert history["val_loss"] == [0.3]

    def test_appends_to_existing_history(self):
        logger = _DummyLogger()
        history = defaultdict(list)
        history["train_loss"].append(0.8)

        BaseLearner.write_metrics_to_logger({"loss": 0.6}, "train", 2, logger, history)

        assert history["train_loss"] == [0.8, 0.6]

    def test_empty_metrics_dict(self):
        logger = _DummyLogger()
        history = defaultdict(list)
        BaseLearner.write_metrics_to_logger({}, "train", 1, logger, history)
        assert len(logger.metrics) == 0


# ===================================================================
# write_lr
# ===================================================================


class TestWriteLR:

    def test_single_param_group(self):
        model = _SimpleModel()
        opt = torch.optim.SGD(model.parameters(), lr=0.05)
        logger = _DummyLogger()
        history = defaultdict(list)

        BaseLearner.write_lr(opt, 1, logger, history)

        assert ("learning_rate", 0.05, 1) in logger.metrics
        assert history["learning_rate"] == [0.05]

    def test_multiple_param_groups(self):
        model = _SimpleModel()
        opt = torch.optim.SGD(
            [
                {"params": model.fc.weight, "lr": 0.01},
                {"params": model.fc.bias, "lr": 0.001},
            ]
        )
        logger = _DummyLogger()
        history = defaultdict(list)

        BaseLearner.write_lr(opt, 3, logger, history)

        assert ("learning_rate/param_group_0", 0.01, 3) in logger.metrics
        assert ("learning_rate/param_group_1", 0.001, 3) in logger.metrics
        assert history["learning_rate/param_group_0"] == [0.01]
        assert history["learning_rate/param_group_1"] == [0.001]

    def test_lr_after_scheduler_step(self):
        model = _SimpleModel()
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.5)
        sched.step()  # lr -> 0.05
        logger = _DummyLogger()
        history = defaultdict(list)

        BaseLearner.write_lr(opt, 1, logger, history)

        assert ("learning_rate", pytest.approx(0.05), 1) in logger.metrics


# ===================================================================
# log_metrics
# ===================================================================


class TestLogMetrics:

    def _make_learner_with_logger(self, tmp_path):
        learner = _make_learner(tmp_path)
        learner.logger = _DummyLogger()
        return learner

    def test_writes_train_and_val(self, tmp_path):
        learner = self._make_learner_with_logger(tmp_path)
        history = defaultdict(list)
        train_metrics = {"loss": 0.5, "acc": 0.8}
        val_metrics = {"loss": 0.4, "acc": 0.85}

        learner.log_metrics(
            val_loader=MagicMock(),
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            metrics_history=history,
            epochs_completed=1,
            logger_img_size=None,
            image_inverse_transform=None,
        )

        tags = [m[0] for m in learner.logger.metrics]
        assert "loss/train" in tags
        assert "acc/train" in tags
        assert "loss/val" in tags
        assert "acc/val" in tags

    def test_writes_predictions_when_img_size_set(self, tmp_path):
        learner = self._make_learner_with_logger(tmp_path)
        learner._task.write_prediction_to_logger = MagicMock()
        val_loader = MagicMock()
        inv_transform = MagicMock()

        learner.log_metrics(
            val_loader=val_loader,
            train_metrics={"loss": 0.5},
            val_metrics={"loss": 0.4},
            metrics_history=defaultdict(list),
            epochs_completed=2,
            logger_img_size=224,
            image_inverse_transform=inv_transform,
        )

        learner._task.write_prediction_to_logger.assert_called_once_with(
            "val",
            val_loader,
            learner.logger,
            inv_transform,
            2,
            img_size=224,
        )

    def test_skips_predictions_when_img_size_none(self, tmp_path):
        learner = self._make_learner_with_logger(tmp_path)
        learner._task.write_prediction_to_logger = MagicMock()

        learner.log_metrics(
            val_loader=MagicMock(),
            train_metrics={"loss": 0.5},
            val_metrics={"loss": 0.4},
            metrics_history=defaultdict(list),
            epochs_completed=1,
            logger_img_size=None,
            image_inverse_transform=None,
        )

        learner._task.write_prediction_to_logger.assert_not_called()

    def test_logger_img_size_tuple(self, tmp_path):
        learner = self._make_learner_with_logger(tmp_path)
        learner._task.write_prediction_to_logger = MagicMock()

        learner.log_metrics(
            val_loader=MagicMock(),
            train_metrics={"loss": 0.5},
            val_metrics={"loss": 0.4},
            metrics_history=defaultdict(list),
            epochs_completed=3,
            logger_img_size=(128, 128),
            image_inverse_transform=None,
        )

        _, kwargs = learner._task.write_prediction_to_logger.call_args
        assert kwargs["img_size"] == (128, 128)


# ===================================================================
# fit / predict raise NotImplementedError
# ===================================================================


class TestAbstractMethods:

    def test_fit_raises(self, tmp_path):
        learner = _make_learner(tmp_path)
        with pytest.raises(NotImplementedError):
            learner.fit()

    def test_predict_raises(self, tmp_path):
        learner = _make_learner(tmp_path)
        with pytest.raises(NotImplementedError):
            learner.predict()
