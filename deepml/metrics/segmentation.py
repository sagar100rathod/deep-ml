from abc import ABC
from typing import Optional, Union

import torch
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryF1Score,
    BinaryJaccardIndex,
    BinaryPrecision,
    BinaryRecall,
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassJaccardIndex,
    MulticlassPrecision,
    MulticlassRecall,
    MultilabelAccuracy,
    MultilabelF1Score,
    MultilabelJaccardIndex,
    MultilabelPrecision,
    MultilabelRecall,
)


class SegmentationMetric(torch.nn.Module, ABC):
    """Abstract base class for segmentation metrics backed by torchmetrics.

    Subclasses assign self._metric (the appropriate torchmetrics instance) in __init__.
    All accumulation and distributed sync is handled by the inner metric.
    """

    is_stateful: bool = (
        True  # signals trainers to bypass SMA and call reset() at epoch start
    )

    def __init__(
        self,
        mode: str,
        reduction: Optional[str],
        num_classes: Optional[int],
        ignore_index: Optional[int],
        threshold: Optional[float],
        class_weights,
        target_class_index: Optional[int],
        zero_division: float,
        activation,
        callable,
    ):
        super().__init__()
        self.mode = mode
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.threshold = threshold
        self.num_classes = num_classes
        self.class_weights = class_weights
        self.zero_division = zero_division
        self.target_class_index = target_class_index
        self.callable_fn = (
            callable  # renamed from self.callable to avoid shadowing builtin
        )
        self._activation = activation or (
            torch.nn.Softmax2d() if mode == "multiclass" else torch.nn.Sigmoid()
        )

        if mode not in ["binary", "multiclass", "multilabel"]:
            raise ValueError(
                "mode should be either 'binary', 'multiclass' or 'multilabel'"
            )
        if ignore_index is not None and mode == "binary":
            raise ValueError("ignore_index is not supported for binary")
        if target_class_index is not None and mode == "binary":
            raise ValueError("target_class_index is not supported for binary")
        if num_classes is None and mode == "multiclass":
            raise ValueError("num_classes is required for multiclass mode")
        if threshold is not None and mode == "multiclass":
            raise ValueError(f"threshold and mode={mode} cannot be used together")
        if (
            target_class_index is not None
            and num_classes is not None
            and target_class_index >= num_classes
        ):
            raise ValueError("target_class_index should be less than num_classes")

    def _torchmetrics_average(self) -> str:
        """Map smp-style reduction string to torchmetrics average parameter.

        imagewise variants are dropped (not supported by torchmetrics) and mapped
        to their non-imagewise equivalent. class_weights and target_class_index
        both require per-class output so force average='none'.
        """
        mapping = {
            "micro": "micro",
            "macro": "macro",
            "weighted": "weighted",
            "macro-imagewise": "macro",
            "micro-imagewise": "micro",
            "weighted-imagewise": "weighted",
            None: "none",
        }
        if self.class_weights is not None or self.target_class_index is not None:
            return "none"
        return mapping.get(self.reduction, "macro")

    def _effective_threshold(self) -> float:
        """Resolve threshold=None → 0.5 for binary/multilabel torchmetrics metrics."""
        return self.threshold if self.threshold is not None else 0.5

    def update(self, output: torch.Tensor, target: torch.Tensor) -> None:
        if self.callable_fn is not None:
            output, target = self.callable_fn(output, target)
        output = self._activation(output)  # always applied after callable
        target = target.to(output.device)
        self._metric.update(output, target)

    def compute(self) -> torch.Tensor:
        result = self._metric.compute()  # scalar or [C] when average='none'
        if self.target_class_index is not None:
            result = result[self.target_class_index]
        elif self.class_weights is not None:
            # Match smp behaviour: multiply by weights then divide by num_classes (not sum of weights)
            w = torch.as_tensor(self.class_weights, dtype=torch.float32).to(
                result.device
            )
            result = (result * w).mean()
        return result

    def reset(self) -> None:
        self._metric.reset()

    def forward(
        self,
        output: Union[torch.LongTensor, torch.FloatTensor],
        target: torch.LongTensor,
    ) -> torch.Tensor:
        self.update(output, target)
        return self.compute()


class Precision(SegmentationMetric):
    """
    Computes the precision metric for segmentation.

    Args:
       mode (str): The mode of the metric, either 'binary', 'multiclass' or 'multilabel'. Default is 'binary'.
       reduction (str, optional): Aggregation mode: 'micro', 'macro', 'weighted'.
           Imagewise variants ('macro-imagewise', etc.) are mapped to their non-imagewise equivalent.
           Default is "macro".
       activation (torch.nn.Module, optional): Activation applied to model output before metric calculation. Default is None.
       ignore_index (int, optional): Target value to ignore. Not supported for binary. Default is None.
       threshold (float, optional): Threshold for binarizing output. Not supported for multiclass. Default is None.
       num_classes (int, optional): Number of classes. Required for multiclass mode. Default is None.
       class_weights: Manual rescaling weights per class. Applied as weighted mean. Default is None.
       zero_division (float): Value returned on zero division. Default is 1.0.
       target_class_index (int, optional): Class index to extract metric for. Not supported for binary. Default is None.
       callable (callable, optional): Preprocessing applied to output and target before activation. Default is None.
    """

    def __init__(
        self,
        mode: str = "binary",
        reduction: Optional[str] = "macro",
        activation=None,
        ignore_index: Optional[int] = None,
        threshold: Optional[float] = None,
        num_classes: Optional[int] = None,
        class_weights=None,
        target_class_index: Optional[int] = None,
        zero_division: float = 1.0,
        callable=None,
    ):
        super().__init__(
            mode=mode,
            reduction=reduction,
            num_classes=num_classes,
            ignore_index=ignore_index,
            threshold=threshold,
            class_weights=class_weights,
            target_class_index=target_class_index,
            zero_division=zero_division,
            activation=activation,
            callable=callable,
        )
        avg = self._torchmetrics_average()
        thr = self._effective_threshold()
        if mode == "binary":
            self._metric = BinaryPrecision(
                threshold=thr, ignore_index=ignore_index, zero_division=zero_division
            )
        elif mode == "multiclass":
            self._metric = MulticlassPrecision(
                num_classes=num_classes,
                average=avg,
                ignore_index=ignore_index,
                zero_division=zero_division,
            )
        else:
            self._metric = MultilabelPrecision(
                num_labels=num_classes,
                threshold=thr,
                average=avg,
                ignore_index=ignore_index,
                zero_division=zero_division,
            )


class Recall(SegmentationMetric):
    """
    Computes the recall metric for segmentation tasks.

    Args:
       mode (str): The mode of the metric, either 'binary', 'multiclass' or 'multilabel'. Default is 'binary'.
       reduction (str, optional): Aggregation mode: 'micro', 'macro', 'weighted'. Default is "macro".
       activation (torch.nn.Module, optional): Activation applied to model output. Default is None.
       ignore_index (int, optional): Target value to ignore. Not supported for binary. Default is None.
       threshold (float, optional): Threshold for binarizing output. Default is None.
       num_classes (int, optional): Number of classes. Required for multiclass. Default is None.
       class_weights: Manual rescaling weights per class. Default is None.
       zero_division (float): Value returned on zero division. Default is 1.0.
       target_class_index (int, optional): Class index to extract metric for. Default is None.
       callable (callable, optional): Preprocessing applied before activation. Default is None.
    """

    def __init__(
        self,
        mode: str = "binary",
        reduction: Optional[str] = "macro",
        activation=None,
        ignore_index: Optional[int] = None,
        threshold: Optional[float] = None,
        num_classes: Optional[int] = None,
        class_weights=None,
        target_class_index: Optional[int] = None,
        zero_division: float = 1.0,
        callable=None,
    ):
        super().__init__(
            mode=mode,
            reduction=reduction,
            num_classes=num_classes,
            ignore_index=ignore_index,
            threshold=threshold,
            class_weights=class_weights,
            target_class_index=target_class_index,
            zero_division=zero_division,
            activation=activation,
            callable=callable,
        )
        avg = self._torchmetrics_average()
        thr = self._effective_threshold()
        if mode == "binary":
            self._metric = BinaryRecall(
                threshold=thr, ignore_index=ignore_index, zero_division=zero_division
            )
        elif mode == "multiclass":
            self._metric = MulticlassRecall(
                num_classes=num_classes,
                average=avg,
                ignore_index=ignore_index,
                zero_division=zero_division,
            )
        else:
            self._metric = MultilabelRecall(
                num_labels=num_classes,
                threshold=thr,
                average=avg,
                ignore_index=ignore_index,
                zero_division=zero_division,
            )


class F1Score(SegmentationMetric):
    """
    Computes the F1 (Dice) metric for segmentation tasks.

    Args:
       mode (str): The mode of the metric, either 'binary', 'multiclass' or 'multilabel'. Default is 'binary'.
       reduction (str, optional): Aggregation mode: 'micro', 'macro', 'weighted'. Default is "macro".
       activation (torch.nn.Module, optional): Activation applied to model output. Default is None.
       ignore_index (int, optional): Target value to ignore. Not supported for binary. Default is None.
       threshold (float, optional): Threshold for binarizing output. Default is None.
       num_classes (int, optional): Number of classes. Required for multiclass. Default is None.
       class_weights: Manual rescaling weights per class. Default is None.
       zero_division (float): Value returned on zero division. Default is 1.0.
       target_class_index (int, optional): Class index to extract metric for. Default is None.
       callable (callable, optional): Preprocessing applied before activation. Default is None.
    """

    def __init__(
        self,
        mode: str = "binary",
        reduction: Optional[str] = "macro",
        activation=None,
        ignore_index: Optional[int] = None,
        threshold: Optional[float] = None,
        num_classes: Optional[int] = None,
        class_weights=None,
        target_class_index: Optional[int] = None,
        zero_division: float = 1.0,
        callable=None,
    ):
        super().__init__(
            mode=mode,
            reduction=reduction,
            num_classes=num_classes,
            ignore_index=ignore_index,
            threshold=threshold,
            class_weights=class_weights,
            target_class_index=target_class_index,
            zero_division=zero_division,
            activation=activation,
            callable=callable,
        )
        avg = self._torchmetrics_average()
        thr = self._effective_threshold()
        if mode == "binary":
            self._metric = BinaryF1Score(
                threshold=thr, ignore_index=ignore_index, zero_division=zero_division
            )
        elif mode == "multiclass":
            self._metric = MulticlassF1Score(
                num_classes=num_classes,
                average=avg,
                ignore_index=ignore_index,
                zero_division=zero_division,
            )
        else:
            self._metric = MultilabelF1Score(
                num_labels=num_classes,
                threshold=thr,
                average=avg,
                ignore_index=ignore_index,
                zero_division=zero_division,
            )


class Accuracy(SegmentationMetric):
    """
    Computes the accuracy metric for segmentation tasks.

    Args:
       mode (str): The mode of the metric, either 'binary', 'multiclass' or 'multilabel'. Default is 'binary'.
       reduction (str, optional): Aggregation mode: 'micro', 'macro', 'weighted'. Default is "macro".
       activation (torch.nn.Module, optional): Activation applied to model output. Default is None.
       ignore_index (int, optional): Target value to ignore. Not supported for binary. Default is None.
       threshold (float, optional): Threshold for binarizing output. Default is None.
       num_classes (int, optional): Number of classes. Required for multiclass. Default is None.
       class_weights: Manual rescaling weights per class. Default is None.
       zero_division (float): Value returned on zero division. Default is 1.0.
       target_class_index (int, optional): Class index to extract metric for. Default is None.
       callable (callable, optional): Preprocessing applied before activation. Default is None.
    """

    def __init__(
        self,
        mode: str = "binary",
        reduction: Optional[str] = "macro",
        activation=None,
        ignore_index: Optional[int] = None,
        threshold: Optional[float] = None,
        num_classes: Optional[int] = None,
        class_weights=None,
        target_class_index: Optional[int] = None,
        zero_division: float = 1.0,
        callable=None,
    ):
        super().__init__(
            mode=mode,
            reduction=reduction,
            num_classes=num_classes,
            ignore_index=ignore_index,
            threshold=threshold,
            class_weights=class_weights,
            target_class_index=target_class_index,
            zero_division=zero_division,
            activation=activation,
            callable=callable,
        )
        avg = self._torchmetrics_average()
        thr = self._effective_threshold()
        if mode == "binary":
            self._metric = BinaryAccuracy(threshold=thr, ignore_index=ignore_index)
        elif mode == "multiclass":
            self._metric = MulticlassAccuracy(
                num_classes=num_classes,
                average=avg,
                ignore_index=ignore_index,
            )
        else:
            self._metric = MultilabelAccuracy(
                num_labels=num_classes,
                threshold=thr,
                average=avg,
                ignore_index=ignore_index,
            )


class IoUScore(SegmentationMetric):
    """
    Computes the Jaccard index (IoU) metric for segmentation.

    Args:
       mode (str): The mode of the metric, either 'binary', 'multiclass' or 'multilabel'. Default is 'binary'.
       reduction (str, optional): Aggregation mode: 'micro', 'macro', 'weighted'. Default is "macro".
       activation (torch.nn.Module, optional): Activation applied to model output. Default is None.
       ignore_index (int, optional): Target value to ignore. Not supported for binary. Default is None.
       threshold (float, optional): Threshold for binarizing output. Default is None.
       num_classes (int, optional): Number of classes. Required for multiclass. Default is None.
       class_weights: Manual rescaling weights per class. Default is None.
       zero_division (float): Value returned on zero division. Default is 1.0.
       target_class_index (int, optional): Class index to extract metric for. Default is None.
       callable (callable, optional): Preprocessing applied before activation. Default is None.
    """

    def __init__(
        self,
        mode: str = "binary",
        reduction: Optional[str] = "macro",
        activation=None,
        ignore_index: Optional[int] = None,
        threshold: Optional[float] = None,
        num_classes: Optional[int] = None,
        class_weights=None,
        target_class_index: Optional[int] = None,
        zero_division: float = 1.0,
        callable=None,
    ):
        super().__init__(
            mode=mode,
            reduction=reduction,
            num_classes=num_classes,
            ignore_index=ignore_index,
            threshold=threshold,
            class_weights=class_weights,
            target_class_index=target_class_index,
            zero_division=zero_division,
            activation=activation,
            callable=callable,
        )
        avg = self._torchmetrics_average()
        thr = self._effective_threshold()
        if mode == "binary":
            self._metric = BinaryJaccardIndex(
                threshold=thr, ignore_index=ignore_index, zero_division=zero_division
            )
        elif mode == "multiclass":
            self._metric = MulticlassJaccardIndex(
                num_classes=num_classes,
                average=avg,
                ignore_index=ignore_index,
                zero_division=zero_division,
            )
        else:
            self._metric = MultilabelJaccardIndex(
                num_labels=num_classes,
                threshold=thr,
                average=avg,
                ignore_index=ignore_index,
                zero_division=zero_division,
            )
