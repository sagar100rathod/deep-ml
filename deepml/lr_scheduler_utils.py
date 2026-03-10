from typing import Literal

from torch.optim.lr_scheduler import OneCycleLR


def setup_one_cycle_lr_scheduler_with_warmup(
    optimizer,
    steps_per_epoch: int,
    warmup_steps: int,
    num_epochs: int = 50,
    max_lr: float = 1e-3,
    anneal_strategy: Literal["cos", "linear"] = "cos",
):
    """
    Returns a lambda function that creates a OneCycleLR scheduler when called with an optimizer.

    :param optimizer: torch.optim.Optimizer
    :param steps_per_epoch: Number of steps in one epoch (usually len(train_loader))
                            If you are using gradient accumulation,
                            this should be the number of steps in one epoch after applying gradient accumulation.
                            If you are using parallel training, this should be the number of steps in one epoch for each process
                            An easy way to calculate this is to use len(train_loader) // gradient_accumulation_steps // num_processes

    :param warmup_steps: Number of warmup steps before the OneCycleLR starts
    :param num_epochs: Total number of epochs for training
    :param max_lr: Maximum learning rate for the OneCycleLR scheduler
    :param anneal_strategy: Annealing strategy for the OneCycleLR scheduler
    :return:
    """

    total_steps = num_epochs * steps_per_epoch
    warmup_ratio = warmup_steps / total_steps

    lr_scheduler = OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=total_steps,
        pct_start=warmup_ratio,
        anneal_strategy=anneal_strategy,
    )

    return lr_scheduler
