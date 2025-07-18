from abc import ABC, abstractmethod
from typing import Union

import torch
from segmentation_models_pytorch.metrics.functional import (
    f1_score,
    get_stats,
    iou_score,
    precision,
    recall,
)


class ToClassIndex(torch.nn.Module):
    def __init__(self, mode: str = "binary", threshold: float = 0.5, activation=None):

        super(ToClassIndex, self).__init__()
        self.mode = mode
        self.activation = activation
        self.threshold = threshold

        if self.mode not in ["binary", "multiclass", "multilabel"]:
            raise ValueError(
                "mode should be either 'binary', 'multiclass' or 'multilabel' "
            )

        if self.threshold and self.mode == "multiclass":
            raise ValueError(f"threshold and mode={self.mode} cannot be used together")

        if self.activation is None:
            self.activation = (
                torch.nn.Softmax2d()
                if self.mode == "multiclass"
                else torch.nn.Sigmoid()
            )

    def forward(self, output: torch.FloatTensor) -> torch.Tensor:

        assert output.ndim == 4  # B,C,H,W

        if self.mode in ["binary", "multilabel"]:
            probability = self.activation(output)
            class_indices = torch.zeros_like(probability)
            class_indices[probability >= self.threshold] = 1
        else:
            # Multiclass
            probability = self.activation(output)
            class_indices = torch.argmax(probability, dim=1)

        return class_indices


class SegmentationMetric(torch.nn.Module, ABC):
    def __init__(
        self,
        mode: str = "binary",
        reduction=None,
        activation=None,
        ignore_index=None,
        threshold=None,
        num_classes=None,
        class_weights=None,
        target_class_index=None,
        zero_division=1.0,
        callable=None,
    ):
        super(SegmentationMetric, self).__init__()
        self.mode = mode
        self.ignore_index = ignore_index
        self.threshold = threshold
        self.num_classes = num_classes
        self.reduction = reduction
        self.class_weights = class_weights
        self.zero_division = zero_division
        self.activation = activation
        self.target_class_index = target_class_index
        self.callable = callable

        if self.mode not in ["binary", "multiclass", "multilabel"]:
            raise ValueError(
                "mode should be either 'binary', 'multiclass' or 'multilabel'"
            )

        if self.ignore_index is not None and self.mode == "binary":
            raise ValueError("ignore_index is not supported for binary")

        if self.target_class_index is not None and self.mode == "binary":
            raise ValueError("target_class_index is not supported for binary")

        if self.num_classes is None and self.mode == "multiclass":
            raise ValueError("num_classes is required for multiclass mode")

        if (
            self.target_class_index is not None
            and self.num_classes is not None
            and self.target_class_index >= self.num_classes
        ):
            raise ValueError("target_class_index should be less than num_classes")

        self.to_class_index = ToClassIndex(self.mode, self.threshold, self.activation)

    @abstractmethod
    def forward(
        self,
        output: Union[torch.LongTensor, torch.FloatTensor],
        target: torch.LongTensor,
    ):
        pass

    def _get_stats(
        self,
        output: Union[torch.LongTensor, torch.FloatTensor],
        target: torch.LongTensor,
    ) -> tuple:

        if self.callable is not None:
            output, target = self.callable(output, target)

        output = self.to_class_index(output)

        if self.mode == "multiclass" and self.ignore_index == 0:
            # to handle class 0 (background) in multiclass segmentation for ignore index
            return get_stats(
                output - 1,
                target - 1,
                ignore_index=-1,
                mode=self.mode,
                num_classes=self.num_classes,
                threshold=self.threshold,
            )
        else:
            return get_stats(
                output,
                target,
                ignore_index=self.ignore_index,
                mode=self.mode,
                num_classes=self.num_classes,
                threshold=self.threshold,
            )


class Precision(SegmentationMetric):
    """
    Computes the precision metric for segmentation.

    Args:
       mode (str): The mode of the metric, either 'binary' or 'multiclass' or 'multilabel'. Default is 'Binary'.
       reduction (str, optional): Define how to aggregate metric between classes and images: 'micro', 'macro', 'weighted'. Default is None.
       activation (torch.nn.Module, optional): An activation function to apply to the output of the model. Default is None.
       ignore_index (int, optional): Specifies a target value that is ignored and does not contribute to the metric calculation. Default is None.
       threshold (float, optional): Threshold value for binarizing the output. Default is None.
       num_classes (int, optional): Number of classes for the metric calculation. Default is None.
       class_weights (torch.Tensor, optional): A manual rescaling weight given to each class. Default is None.
       zero_division (float): Value to return when there is a zero division. Default is 1.0.
       target_class_index (int, optional): The class index for which to compute the precision. Default is None.
       callabe (callable, optional): A callable function to apply to the output and target before metric calculation. Default is None.
    """

    def __init__(
        self,
        mode: str = "binary",
        reduction=None,
        activation=None,
        ignore_index=None,
        threshold=None,
        num_classes=None,
        class_weights=None,
        target_class_index=None,
        zero_division=1.0,
        callable=None,
    ):
        super(Precision, self).__init__(
            mode=mode,
            reduction=reduction,
            activation=activation,
            ignore_index=ignore_index,
            threshold=threshold,
            num_classes=num_classes,
            class_weights=class_weights,
            target_class_index=target_class_index,
            zero_division=zero_division,
            callable=callable,
        )

    def forward(
        self,
        output: Union[torch.LongTensor, torch.FloatTensor],
        target: torch.LongTensor,
    ):

        tp, fp, fn, tn = self._get_stats(output, target)

        # tp shape is [N, C] where N is the batch size and C is the number of classes
        # for each image in the batch, we have tp, fp, fn, tn for each class

        if self.target_class_index is not None:
            tp = tp[:, self.target_class_index]
            fp = fp[:, self.target_class_index]
            fn = fn[:, self.target_class_index]
            tn = tn[:, self.target_class_index]

        return precision(
            tp=tp,
            fp=fp,
            fn=fn,
            tn=tn,
            reduction=self.reduction,
            class_weights=self.class_weights,
            zero_division=self.zero_division,
        )


class Recall(SegmentationMetric):
    """
    Computes the recall metric for segmentation tasks.

    Args:
       mode (str): The mode of the metric, either 'binary' or 'multiclass' or 'multilabel'. Default is 'Binary'.
       reduction (str, optional): Define how to aggregate metric between classes and images: 'micro', 'macro', 'weighted'. Default is None.
       activation (torch.nn.Module, optional): An activation function to apply to the output of the model. Default is None.
       ignore_index (int, optional): Specifies a target value that is ignored and does not contribute to the metric calculation. Default is None.
       threshold (float, optional): Threshold value for binarizing the output. Default is None.
       num_classes (int, optional): Number of classes for the metric calculation. Default is None.
       class_weights (torch.Tensor, optional): A manual rescaling weight given to each class. Default is None.
       target_class_index (int, optional): The class index for which to compute the recall. Default is None.
       zero_division (float): Value to return when there is a zero division. Default is 1.0.
       callable (callable, optional): A callable function to apply to the output and target before metric calculation. Default is None.
    """

    def __init__(
        self,
        mode: str = "binary",
        reduction=None,
        activation=None,
        ignore_index=None,
        threshold=None,
        num_classes=None,
        class_weights=None,
        target_class_index=None,
        zero_division=1.0,
        callable=None,
    ):
        super(Recall, self).__init__(
            mode=mode,
            reduction=reduction,
            activation=activation,
            ignore_index=ignore_index,
            threshold=threshold,
            num_classes=num_classes,
            class_weights=class_weights,
            target_class_index=target_class_index,
            zero_division=zero_division,
            callable=callable,
        )

    def forward(
        self,
        output: Union[torch.LongTensor, torch.FloatTensor],
        target: torch.LongTensor,
    ):
        tp, fp, fn, tn = self._get_stats(output, target)

        # tp shape is [N, C] where N is the batch size and C is the number of classes
        # for each image in the batch, we have tp, fp, fn, tn for each class

        if self.target_class_index is not None:
            tp = tp[:, self.target_class_index]
            fp = fp[:, self.target_class_index]
            fn = fn[:, self.target_class_index]
            tn = tn[:, self.target_class_index]

        return recall(
            tp=tp,
            fp=fp,
            fn=fn,
            tn=tn,
            reduction=self.reduction,
            class_weights=self.class_weights,
            zero_division=self.zero_division,
        )


class F1Score(SegmentationMetric):
    """
    Computes the f1 metric for segmentation tasks.

    Args:
       mode (str): The mode of the metric, either 'binary' or 'multiclass' or 'multilabel'. Default is 'Binary'.
       reduction (str, optional): Define how to aggregate metric between classes and images: 'micro', 'macro', 'weighted'. Default is None.
       activation (torch.nn.Module, optional): An activation function to apply to the output of the model. Default is None.
       ignore_index (int, optional): Specifies a target value that is ignored and does not contribute to the metric calculation. Default is None.
       threshold (float, optional): Threshold value for binarizing the output. Default is None.
       num_classes (int, optional): Number of classes for the metric calculation. Default is None.
       class_weights (torch.Tensor, optional): A manual rescaling weight given to each class. Default is None.
       target_class_index (int, optional): The class index for which to compute the f1 score. Default is None.
       zero_division (float): Value to return when there is a zero division. Default is 1.0.
       callable (callable, optional): A callable function to apply to the output and target before metric calculation. Default is None.
    """

    def __init__(
        self,
        mode: str = "binary",
        reduction=None,
        activation=None,
        ignore_index=None,
        threshold=None,
        num_classes=None,
        class_weights=None,
        target_class_index=None,
        zero_division=1.0,
        callable=None,
    ):
        super(F1Score, self).__init__(
            mode=mode,
            reduction=reduction,
            activation=activation,
            ignore_index=ignore_index,
            threshold=threshold,
            num_classes=num_classes,
            class_weights=class_weights,
            target_class_index=target_class_index,
            zero_division=zero_division,
            callable=callable,
        )

    def forward(
        self,
        output: Union[torch.LongTensor, torch.FloatTensor],
        target: torch.LongTensor,
    ):
        tp, fp, fn, tn = self._get_stats(output, target)

        # tp shape is [N, C] where N is the batch size and C is the number of classes
        # for each image in the batch, we have tp, fp, fn, tn for each class

        if self.target_class_index is not None:
            tp = tp[:, self.target_class_index]
            fp = fp[:, self.target_class_index]
            fn = fn[:, self.target_class_index]
            tn = tn[:, self.target_class_index]

        return f1_score(
            tp=tp,
            fp=fp,
            fn=fn,
            tn=tn,
            reduction=self.reduction,
            class_weights=self.class_weights,
            zero_division=self.zero_division,
        )


class IoUScore(SegmentationMetric):
    """
    Computes the jaccard index metric for segmentation.

    Args:
       mode (str): The mode of the metric, either 'binary' or 'multiclass' or 'multilabel'. Default is 'Binary'.
       reduction (str, optional): Define how to aggregate metric between classes and images: 'micro', 'macro', 'weighted'. Default is None.
       activation (torch.nn.Module, optional): An activation function to apply to the output of the model. Default is None.
       ignore_index (int, optional): Specifies a target value that is ignored and does not contribute to the metric calculation. Default is None.
       threshold (float, optional): Threshold value for binarizing the output. Default is None.
       num_classes (int, optional): Number of classes for the metric calculation. Default is None.
       class_weights (torch.Tensor, optional): A manual rescaling weight given to each class. Default is None.
       zero_division (float): Value to return when there is a zero division. Default is 1.0.
       target_class_index (int, optional): The class index for which to compute the precision. Default is None.
       callable (callable, optional): A callable function to apply to the output and target before metric calculation. Default is None.
    """

    def __init__(
        self,
        mode: str = "binary",
        reduction=None,
        activation=None,
        ignore_index=None,
        threshold=None,
        num_classes=None,
        class_weights=None,
        target_class_index=None,
        zero_division=1.0,
        callable=None,
    ):
        super(IoUScore, self).__init__(
            mode=mode,
            reduction=reduction,
            activation=activation,
            ignore_index=ignore_index,
            threshold=threshold,
            num_classes=num_classes,
            class_weights=class_weights,
            target_class_index=target_class_index,
            zero_division=zero_division,
            callable=callable,
        )

    def forward(
        self,
        output: Union[torch.LongTensor, torch.FloatTensor],
        target: torch.LongTensor,
    ):
        tp, fp, fn, tn = self._get_stats(output, target)

        # tp shape is [N, C] where N is the batch size and C is the number of classes
        # for each image in the batch, we have tp, fp, fn, tn for each class

        if self.target_class_index is not None:
            tp = tp[:, self.target_class_index]
            fp = fp[:, self.target_class_index]
            fn = fn[:, self.target_class_index]
            tn = tn[:, self.target_class_index]

        return iou_score(
            tp=tp,
            fp=fp,
            fn=fn,
            tn=tn,
            reduction=self.reduction,
            class_weights=self.class_weights,
            zero_division=self.zero_division,
        )
