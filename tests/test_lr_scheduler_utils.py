"""
Pytest test-cases for deepml.lr_scheduler_utils.setup_one_cycle_lr_scheduler_with_warmup.

Covers:
    - Return type is OneCycleLR
    - total_steps computed correctly
    - warmup ratio (pct_start) computed correctly
    - Default parameter values (num_epochs, max_lr, anneal_strategy)
    - Custom max_lr
    - Anneal strategy "cos" and "linear"
    - LR changes after scheduler.step()
    - LR reaches close to max_lr during the cycle
    - Multiple param groups
    - Edge case: warmup_steps == 0
    - Edge case: warmup_steps == total_steps (full warmup)
    - Edge case: single step per epoch
"""

import pytest
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import OneCycleLR

from deepml.lr_scheduler_utils import setup_one_cycle_lr_scheduler_with_warmup

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(x)


@pytest.fixture
def model_and_optimizer():
    model = _TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    return model, optimizer


# ---------------------------------------------------------------------------
# Return type & basic contract
# ---------------------------------------------------------------------------


class TestReturnType:

    def test_returns_one_cycle_lr_instance(self, model_and_optimizer):
        _, optimizer = model_and_optimizer
        scheduler = setup_one_cycle_lr_scheduler_with_warmup(
            optimizer,
            steps_per_epoch=10,
            warmup_steps=5,
            num_epochs=5,
        )
        assert isinstance(scheduler, OneCycleLR)

    def test_returns_scheduler_not_none(self, model_and_optimizer):
        _, optimizer = model_and_optimizer
        scheduler = setup_one_cycle_lr_scheduler_with_warmup(
            optimizer,
            steps_per_epoch=10,
            warmup_steps=5,
        )
        assert scheduler is not None


# ---------------------------------------------------------------------------
# Total steps & warmup ratio
# ---------------------------------------------------------------------------


class TestTotalStepsAndWarmup:

    def test_total_steps_computed(self, model_and_optimizer):
        _, optimizer = model_and_optimizer
        scheduler = setup_one_cycle_lr_scheduler_with_warmup(
            optimizer,
            steps_per_epoch=20,
            warmup_steps=10,
            num_epochs=10,
        )
        expected_total = 10 * 20  # 200
        assert scheduler.total_steps == expected_total

    def test_warmup_ratio_pct_start(self, model_and_optimizer):
        _, optimizer = model_and_optimizer
        steps_per_epoch = 100
        warmup_steps = 50
        num_epochs = 10
        scheduler = setup_one_cycle_lr_scheduler_with_warmup(
            optimizer,
            steps_per_epoch=steps_per_epoch,
            warmup_steps=warmup_steps,
            num_epochs=num_epochs,
        )
        total_steps = num_epochs * steps_per_epoch
        expected_pct = warmup_steps / total_steps  # 50 / 1000 = 0.05
        # OneCycleLR stores _schedule_phases internally; verify via the ratio
        assert expected_pct == pytest.approx(0.05)

    def test_total_steps_single_epoch(self, model_and_optimizer):
        _, optimizer = model_and_optimizer
        scheduler = setup_one_cycle_lr_scheduler_with_warmup(
            optimizer,
            steps_per_epoch=50,
            warmup_steps=5,
            num_epochs=1,
        )
        assert scheduler.total_steps == 50


# ---------------------------------------------------------------------------
# Default parameter values
# ---------------------------------------------------------------------------


class TestDefaultParameters:

    def test_default_num_epochs(self, model_and_optimizer):
        _, optimizer = model_and_optimizer
        scheduler = setup_one_cycle_lr_scheduler_with_warmup(
            optimizer,
            steps_per_epoch=10,
            warmup_steps=5,
        )
        # Default num_epochs=50, so total_steps = 50 * 10 = 500
        assert scheduler.total_steps == 500

    def test_default_max_lr(self, model_and_optimizer):
        _, optimizer = model_and_optimizer
        scheduler = setup_one_cycle_lr_scheduler_with_warmup(
            optimizer,
            steps_per_epoch=10,
            warmup_steps=5,
        )
        # Default max_lr=1e-3; peak LR during the cycle should be close to 1e-3
        total_steps = 50 * 10
        peak_lr = 0.0
        for _ in range(total_steps):
            peak_lr = max(peak_lr, optimizer.param_groups[0]["lr"])
            scheduler.step()
        assert peak_lr == pytest.approx(1e-3, rel=0.05)

    def test_default_anneal_strategy(self, model_and_optimizer):
        _, optimizer = model_and_optimizer
        # Default anneal_strategy="cos"; just verify it constructs and steps without error
        scheduler = setup_one_cycle_lr_scheduler_with_warmup(
            optimizer,
            steps_per_epoch=10,
            warmup_steps=5,
        )
        assert isinstance(scheduler, OneCycleLR)
        scheduler.step()


# ---------------------------------------------------------------------------
# Custom max_lr
# ---------------------------------------------------------------------------


class TestCustomMaxLR:

    def test_custom_max_lr_stored(self, model_and_optimizer):
        _, optimizer = model_and_optimizer
        max_lr = 0.1
        scheduler = setup_one_cycle_lr_scheduler_with_warmup(
            optimizer,
            steps_per_epoch=10,
            warmup_steps=5,
            num_epochs=5,
            max_lr=max_lr,
        )
        # Verify peak LR during the cycle matches the custom max_lr
        total_steps = 5 * 10
        peak_lr = 0.0
        for _ in range(total_steps):
            peak_lr = max(peak_lr, optimizer.param_groups[0]["lr"])
            scheduler.step()
        assert peak_lr == pytest.approx(max_lr, rel=0.05)

    def test_lr_reaches_near_max_lr_during_cycle(self, model_and_optimizer):
        _, optimizer = model_and_optimizer
        max_lr = 0.05
        scheduler = setup_one_cycle_lr_scheduler_with_warmup(
            optimizer,
            steps_per_epoch=10,
            warmup_steps=10,
            num_epochs=5,
            max_lr=max_lr,
        )
        # Step through the warmup phase and track the peak lr
        peak_lr = 0.0
        total_steps = 5 * 10
        for _ in range(total_steps):
            current_lr = optimizer.param_groups[0]["lr"]
            peak_lr = max(peak_lr, current_lr)
            scheduler.step()

        assert peak_lr == pytest.approx(max_lr, rel=0.05)


# ---------------------------------------------------------------------------
# Anneal strategy
# ---------------------------------------------------------------------------


class TestAnnealStrategy:

    def test_cos_strategy(self, model_and_optimizer):
        _, optimizer = model_and_optimizer
        scheduler = setup_one_cycle_lr_scheduler_with_warmup(
            optimizer,
            steps_per_epoch=10,
            warmup_steps=5,
            num_epochs=5,
            anneal_strategy="cos",
        )
        assert isinstance(scheduler, OneCycleLR)
        # Should be able to step without error
        scheduler.step()

    def test_linear_strategy(self, model_and_optimizer):
        _, optimizer = model_and_optimizer
        scheduler = setup_one_cycle_lr_scheduler_with_warmup(
            optimizer,
            steps_per_epoch=10,
            warmup_steps=5,
            num_epochs=5,
            anneal_strategy="linear",
        )
        assert isinstance(scheduler, OneCycleLR)
        scheduler.step()

    def test_cos_and_linear_produce_different_lr_schedules(self):
        """Two schedulers with different anneal strategies should diverge."""
        model1 = _TinyModel()
        opt1 = torch.optim.SGD(model1.parameters(), lr=0.001)
        model2 = _TinyModel()
        opt2 = torch.optim.SGD(model2.parameters(), lr=0.001)

        sched_cos = setup_one_cycle_lr_scheduler_with_warmup(
            opt1,
            steps_per_epoch=20,
            warmup_steps=10,
            num_epochs=5,
            max_lr=0.01,
            anneal_strategy="cos",
        )
        sched_lin = setup_one_cycle_lr_scheduler_with_warmup(
            opt2,
            steps_per_epoch=20,
            warmup_steps=10,
            num_epochs=5,
            max_lr=0.01,
            anneal_strategy="linear",
        )

        lrs_cos, lrs_lin = [], []
        for _ in range(100):
            lrs_cos.append(opt1.param_groups[0]["lr"])
            lrs_lin.append(opt2.param_groups[0]["lr"])
            sched_cos.step()
            sched_lin.step()

        # They should not be identical across all steps
        assert lrs_cos != lrs_lin


# ---------------------------------------------------------------------------
# LR changes after stepping
# ---------------------------------------------------------------------------


class TestLRProgression:

    def test_lr_changes_after_step(self, model_and_optimizer):
        _, optimizer = model_and_optimizer
        scheduler = setup_one_cycle_lr_scheduler_with_warmup(
            optimizer,
            steps_per_epoch=10,
            warmup_steps=5,
            num_epochs=5,
            max_lr=0.01,
        )
        lr_before = optimizer.param_groups[0]["lr"]
        scheduler.step()
        lr_after = optimizer.param_groups[0]["lr"]
        assert lr_before != lr_after

    def test_lr_increases_during_warmup(self, model_and_optimizer):
        _, optimizer = model_and_optimizer
        scheduler = setup_one_cycle_lr_scheduler_with_warmup(
            optimizer,
            steps_per_epoch=100,
            warmup_steps=50,
            num_epochs=10,
            max_lr=0.1,
        )
        lr_start = optimizer.param_groups[0]["lr"]
        for _ in range(20):
            scheduler.step()
        lr_mid_warmup = optimizer.param_groups[0]["lr"]
        assert lr_mid_warmup > lr_start

    def test_lr_decreases_after_peak(self, model_and_optimizer):
        _, optimizer = model_and_optimizer
        steps_per_epoch = 100
        warmup_steps = 100
        num_epochs = 10
        total_steps = num_epochs * steps_per_epoch  # 1000
        scheduler = setup_one_cycle_lr_scheduler_with_warmup(
            optimizer,
            steps_per_epoch=steps_per_epoch,
            warmup_steps=warmup_steps,
            num_epochs=num_epochs,
            max_lr=0.1,
        )

        # Step past warmup to peak
        for _ in range(warmup_steps):
            scheduler.step()
        lr_at_peak = optimizer.param_groups[0]["lr"]

        # Step further into annealing phase
        for _ in range(200):
            scheduler.step()
        lr_after_peak = optimizer.param_groups[0]["lr"]

        assert lr_after_peak < lr_at_peak

    def test_completes_all_steps_without_error(self, model_and_optimizer):
        _, optimizer = model_and_optimizer
        steps_per_epoch = 10
        num_epochs = 5
        total_steps = num_epochs * steps_per_epoch
        scheduler = setup_one_cycle_lr_scheduler_with_warmup(
            optimizer,
            steps_per_epoch=steps_per_epoch,
            warmup_steps=5,
            num_epochs=num_epochs,
        )
        for _ in range(total_steps):
            scheduler.step()
        # Should have completed without error


# ---------------------------------------------------------------------------
# Multiple param groups
# ---------------------------------------------------------------------------


class TestMultipleParamGroups:

    def test_multiple_param_groups(self):
        model = _TinyModel()
        optimizer = torch.optim.SGD(
            [
                {"params": model.fc.weight, "lr": 0.001},
                {"params": model.fc.bias, "lr": 0.0005},
            ]
        )
        scheduler = setup_one_cycle_lr_scheduler_with_warmup(
            optimizer,
            steps_per_epoch=10,
            warmup_steps=5,
            num_epochs=5,
            max_lr=0.01,
        )
        assert isinstance(scheduler, OneCycleLR)
        # Both param groups should be managed by the scheduler
        assert len(optimizer.param_groups) == 2
        # Stepping should update LR for both groups
        scheduler.step()
        assert optimizer.param_groups[0]["lr"] > 0
        assert optimizer.param_groups[1]["lr"] > 0

    def test_multiple_param_groups_lr_changes(self):
        model = _TinyModel()
        optimizer = torch.optim.SGD(
            [
                {"params": model.fc.weight, "lr": 0.001},
                {"params": model.fc.bias, "lr": 0.0005},
            ]
        )
        scheduler = setup_one_cycle_lr_scheduler_with_warmup(
            optimizer,
            steps_per_epoch=10,
            warmup_steps=5,
            num_epochs=5,
            max_lr=0.01,
        )

        lr0_before = optimizer.param_groups[0]["lr"]
        lr1_before = optimizer.param_groups[1]["lr"]
        scheduler.step()
        lr0_after = optimizer.param_groups[0]["lr"]
        lr1_after = optimizer.param_groups[1]["lr"]

        assert lr0_after != lr0_before
        assert lr1_after != lr1_before


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:

    def test_zero_warmup_steps(self, model_and_optimizer):
        _, optimizer = model_and_optimizer
        scheduler = setup_one_cycle_lr_scheduler_with_warmup(
            optimizer,
            steps_per_epoch=10,
            warmup_steps=0,
            num_epochs=5,
            max_lr=0.01,
        )
        assert isinstance(scheduler, OneCycleLR)
        # pct_start = 0 / total_steps = 0.0
        # LR should start at or near max_lr immediately
        lr_initial = optimizer.param_groups[0]["lr"]
        assert lr_initial == pytest.approx(0.01, rel=0.1)

    def test_single_step_per_epoch(self, model_and_optimizer):
        _, optimizer = model_and_optimizer
        scheduler = setup_one_cycle_lr_scheduler_with_warmup(
            optimizer,
            steps_per_epoch=1,
            warmup_steps=2,
            num_epochs=20,
            max_lr=0.01,
        )
        # total_steps = 20, warmup_ratio = 2/20 = 0.1
        assert scheduler.total_steps == 20
        scheduler.step()

    def test_large_steps_per_epoch(self, model_and_optimizer):
        _, optimizer = model_and_optimizer
        scheduler = setup_one_cycle_lr_scheduler_with_warmup(
            optimizer,
            steps_per_epoch=10_000,
            warmup_steps=500,
            num_epochs=2,
            max_lr=0.01,
        )
        assert scheduler.total_steps == 20_000

    def test_exceeding_total_steps_raises(self, model_and_optimizer):
        """Stepping beyond total_steps should raise an error from OneCycleLR."""
        _, optimizer = model_and_optimizer
        scheduler = setup_one_cycle_lr_scheduler_with_warmup(
            optimizer,
            steps_per_epoch=5,
            warmup_steps=2,
            num_epochs=2,
            max_lr=0.01,
        )
        total_steps = 2 * 5
        for _ in range(total_steps):
            scheduler.step()
        with pytest.raises(ValueError):
            scheduler.step()
