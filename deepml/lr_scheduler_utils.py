from typing import Literal, Optional

from torch.optim.lr_scheduler import OneCycleLR


def setup_one_cycle_lr_scheduler_with_warmup(
    optimizer,
    steps_per_epoch: int,
    warmup_steps: Optional[int] = None,
    warmup_ratio: Optional[float] = None,
    num_epochs: int = 50,
    max_lr: float = 1e-3,
    anneal_strategy: Literal["cos", "linear"] = "cos",
):
    """Sets up a OneCycleLR learning rate scheduler with warmup phase.

    Creates a OneCycleLR scheduler that includes a warmup phase specified either
    by the number of steps or as a ratio of total training steps. The scheduler
    follows the 1-cycle policy: warmup → annealing to max_lr → annealing to min_lr.

    Args:
        optimizer: PyTorch optimizer instance to schedule.
        steps_per_epoch: Number of optimizer steps in one epoch. Typically
            ``len(train_loader)``. When using gradient accumulation or distributed
            training, adjust accordingly:
            ``len(train_loader) // gradient_accumulation_steps // num_processes``.
        warmup_steps: Number of warmup steps before reaching max_lr. Must be less
            than total training steps. Mutually exclusive with warmup_ratio.
            Defaults to None.
        warmup_ratio: Ratio of total training steps to use for warmup (0-1).
            Mutually exclusive with warmup_steps. Defaults to None.
        num_epochs: Total number of training epochs. Defaults to 50.
        max_lr: Maximum learning rate during the cycle. Defaults to 1e-3.
        anneal_strategy: Annealing strategy after warmup. Options:
            - ``"cos"``: Cosine annealing (smooth decay)
            - ``"linear"``: Linear annealing
            Defaults to ``"cos"``.

    Returns:
        OneCycleLR scheduler instance configured with the specified parameters.

    Raises:
        AssertionError: If neither warmup_steps nor warmup_ratio is provided.
        AssertionError: If both warmup_steps and warmup_ratio are provided.
        AssertionError: If warmup_steps >= total training steps.
        AssertionError: If warmup_ratio is not between 0 and 1.

    Example:
        >>> from torch.optim import Adam
        >>> optimizer = Adam(model.parameters(), lr=1e-4)
        >>> scheduler = setup_one_cycle_lr_scheduler_with_warmup(
        ...     optimizer,
        ...     steps_per_epoch=100,
        ...     warmup_ratio=0.1,
        ...     num_epochs=50,
        ...     max_lr=1e-3
        ... )

    Note:
        The OneCycleLR policy divides training into three phases:
        1. Warmup: Learning rate increases from initial_lr to max_lr
        2. Annealing: Learning rate decreases from max_lr towards min_lr
        3. The pct_start parameter controls the fraction of total steps for warmup
    """

    # Validate that exactly one of warmup_steps or warmup_ratio is provided
    assert (warmup_steps is None) != (
        warmup_ratio is None
    ), "Exactly one of warmup_steps or warmup_ratio must be provided, not both or neither."

    total_steps = num_epochs * steps_per_epoch

    if warmup_steps is not None:
        assert (
            warmup_steps < total_steps
        ), f"Warmup steps ({warmup_steps}) must be less than total training steps ({total_steps})"

        warmup_ratio = warmup_steps / total_steps
    else:
        # warmup_ratio is provided
        assert (
            0 < warmup_ratio < 1
        ), f"Warmup ratio ({warmup_ratio}) must be between 0 and 1"

    lr_scheduler = OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=total_steps,
        pct_start=warmup_ratio,
        anneal_strategy=anneal_strategy,
    )

    return lr_scheduler
