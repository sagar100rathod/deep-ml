from typing import Optional

import torch
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryFBetaScore,
    BinaryMatthewsCorrCoef,
    BinaryPrecision,
    BinaryRecall,
    MulticlassAccuracy,
    MulticlassFBetaScore,
    MulticlassMatthewsCorrCoef,
    MulticlassPrecision,
    MulticlassRecall,
)


def _to_long(target: torch.Tensor) -> torch.Tensor:
    return target.long() if target.is_floating_point() else target


def _check_multiclass_shape(output: torch.Tensor, num_classes: Optional[int]) -> None:
    if num_classes is None and output.dim() == 2 and output.shape[-1] > 1:
        raise ValueError(
            f"Received predictions of shape {tuple(output.shape)}, which suggests "
            f"multiclass input with {output.shape[-1]} classes. "
            f"Initialize with num_classes={output.shape[-1]} for multiclass classification."
        )


class Accuracy(torch.nn.Module):
    """Image-level accuracy metric.

    Args:
        num_classes: Number of classes. None → binary, int → multiclass.
        threshold: Decision threshold for binary classification. Default 0.5.
    """

    is_stateful: bool = True

    def __init__(self, num_classes: Optional[int] = None, threshold: float = 0.5):
        super().__init__()
        self._num_classes = num_classes
        if num_classes is None:
            self._metric = BinaryAccuracy(threshold=threshold)
        else:
            self._metric = MulticlassAccuracy(num_classes=num_classes, average="macro")

    def update(self, output: torch.Tensor, target: torch.Tensor) -> None:
        _check_multiclass_shape(output, self._num_classes)
        self._metric.update(output, _to_long(target))

    def compute(self) -> torch.Tensor:
        return self._metric.compute()

    def reset(self) -> None:
        self._metric.reset()

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        self.update(output, target)
        return self.compute()


class Precision(torch.nn.Module):
    """Image-level precision metric.

    Args:
        num_classes: Number of classes. None → binary, int → multiclass.
        threshold: Decision threshold for binary classification. Default 0.5.
        zero_division: Value returned on zero division. Default 0.0.
    """

    is_stateful: bool = True

    def __init__(
        self,
        num_classes: Optional[int] = None,
        threshold: float = 0.5,
        zero_division: float = 0.0,
    ):
        super().__init__()
        self._num_classes = num_classes
        if num_classes is None:
            self._metric = BinaryPrecision(
                threshold=threshold, zero_division=zero_division
            )
        else:
            self._metric = MulticlassPrecision(
                num_classes=num_classes, average="macro", zero_division=zero_division
            )

    def update(self, output: torch.Tensor, target: torch.Tensor) -> None:
        _check_multiclass_shape(output, self._num_classes)
        self._metric.update(output, _to_long(target))

    def compute(self) -> torch.Tensor:
        return self._metric.compute()

    def reset(self) -> None:
        self._metric.reset()

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        self.update(output, target)
        return self.compute()


class Recall(torch.nn.Module):
    """Image-level recall metric.

    Args:
        num_classes: Number of classes. None → binary, int → multiclass.
        threshold: Decision threshold for binary classification. Default 0.5.
        zero_division: Value returned on zero division. Default 0.0.
    """

    is_stateful: bool = True

    def __init__(
        self,
        num_classes: Optional[int] = None,
        threshold: float = 0.5,
        zero_division: float = 0.0,
    ):
        super().__init__()
        self._num_classes = num_classes
        if num_classes is None:
            self._metric = BinaryRecall(
                threshold=threshold, zero_division=zero_division
            )
        else:
            self._metric = MulticlassRecall(
                num_classes=num_classes, average="macro", zero_division=zero_division
            )

    def update(self, output: torch.Tensor, target: torch.Tensor) -> None:
        _check_multiclass_shape(output, self._num_classes)
        self._metric.update(output, _to_long(target))

    def compute(self) -> torch.Tensor:
        return self._metric.compute()

    def reset(self) -> None:
        self._metric.reset()

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        self.update(output, target)
        return self.compute()


class FScore(torch.nn.Module):
    """Image-level F-beta score.

    Args:
        num_classes: Number of classes. None → binary, int → multiclass.
        beta: Beta factor. beta=1 → F1, beta=2 → F2 (recall-weighted). Default 1.0.
        threshold: Decision threshold for binary classification. Default 0.5.
        zero_division: Value returned on zero division. Default 0.0.
    """

    is_stateful: bool = True

    def __init__(
        self,
        num_classes: Optional[int] = None,
        beta: float = 1.0,
        threshold: float = 0.5,
        zero_division: float = 0.0,
    ):
        super().__init__()
        self._num_classes = num_classes
        if num_classes is None:
            self._metric = BinaryFBetaScore(
                beta=beta, threshold=threshold, zero_division=zero_division
            )
        else:
            self._metric = MulticlassFBetaScore(
                num_classes=num_classes,
                beta=beta,
                average="macro",
                zero_division=zero_division,
            )

    def update(self, output: torch.Tensor, target: torch.Tensor) -> None:
        _check_multiclass_shape(output, self._num_classes)
        self._metric.update(output, _to_long(target))

    def compute(self) -> torch.Tensor:
        return self._metric.compute()

    def reset(self) -> None:
        self._metric.reset()

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        self.update(output, target)
        return self.compute()


class MCC(torch.nn.Module):
    """Matthews Correlation Coefficient — useful for imbalanced datasets.

    Args:
        num_classes: Number of classes. None → binary, int → multiclass.
        threshold: Decision threshold for binary classification. Default 0.5.
    """

    is_stateful: bool = True

    def __init__(self, num_classes: Optional[int] = None, threshold: float = 0.5):
        super().__init__()
        self._num_classes = num_classes
        if num_classes is None:
            self._metric = BinaryMatthewsCorrCoef(threshold=threshold)
        else:
            self._metric = MulticlassMatthewsCorrCoef(num_classes=num_classes)

    def update(self, output: torch.Tensor, target: torch.Tensor) -> None:
        _check_multiclass_shape(output, self._num_classes)
        self._metric.update(output, _to_long(target))

    def compute(self) -> torch.Tensor:
        return self._metric.compute()

    def reset(self) -> None:
        self._metric.reset()

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        self.update(output, target)
        return self.compute()
