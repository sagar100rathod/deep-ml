from abc import ABC, abstractmethod
from typing import Union
import torch
from segmentation_models_pytorch.metrics.functional import (
    get_stats,
    precision,
    recall,
    f1_score,
)


class Binarizer(torch.nn.Module):
    def __init__(self, threshold=0.5, activation=None, value=1):
        super(Binarizer, self).__init__()
        self.activation = activation
        self.threshold = threshold
        self.value = value

    def forward(self, output: torch.FloatTensor):
        if self.activation is not None:
            output = self.activation(output)

        output[output >= self.threshold] = self.value
        output[output < self.threshold] = 0

        return output.to(torch.uint8)


class Accuracy(torch.nn.Module):
    def __init__(self, is_multiclass=False, threshold=0.5):
        super(Accuracy, self).__init__()

        if is_multiclass:
            self.activation = torch.nn.Softmax2d()
        else:
            self.activation = Binarizer(threshold, torch.nn.Sigmoid())

    def forward(self, output, target):

        output = self.activation(output)

        output = output.to(torch.float)
        target = target.to(torch.float)

        return (output == target).float().mean()


class SegmentationMetric(torch.nn.Module, ABC):
    def __init__(
        self,
        mode: str = "binary",
        reduction=None,
        from_logits=True,
        activation=None,
        ignore_index=None,
        threshold=None,
        num_classes=None,
        class_weights=None,
        zero_division=1.0,

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

        if self.mode not in ["binary", "multiclass", "multilabel"]:
            raise ValueError("mode should be either 'binary', 'multiclass' or 'multilabel'")

        if self.ignore_index is not None and self.mode == "binary":
            raise ValueError("ignore_index is not supported for binary")

        if from_logits and self.activation is not None:
            raise ValueError("from_logits and activation cannot be used together")

        if from_logits and self.activation is None:
            self.activation = torch.nn.Softmax2d() if self.mode == "multiclass" else torch.nn.Sigmoid()

    @abstractmethod
    def forward(
        self,
        output: Union[torch.LongTensor, torch.FloatTensor],
        target: torch.LongTensor,
    ):
        pass

    def _get_stats(self, output: Union[torch.LongTensor, torch.FloatTensor],
                        target: torch.LongTensor) -> tuple:

        if self.activation is not None:
            output = self.activation(output)

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
         from_logits (bool, optional): If True, the input is expected to be the raw output of a model. Default is True.
         activation (torch.nn.Module, optional): An activation function to apply to the output of the model. Default is None.
         ignore_index (int, optional): Specifies a target value that is ignored and does not contribute to the metric calculation. Default is None.
         threshold (float, optional): Threshold value for binarizing the output. Default is None.
         num_classes (int, optional): Number of classes for the metric calculation. Default is None.
         class_weights (torch.Tensor, optional): A manual rescaling weight given to each class. Default is None.
         zero_division (float): Value to return when there is a zero division. Default is 1.0.
    """
    def __init__(
        self,
        mode: str = "binary",
        reduction=None,
        from_logits=True,
        activation=None,
        ignore_index=None,
        threshold=None,
        num_classes=None,
        class_weights=None,
        zero_division=1.0,
    ):
        super(Precision, self).__init__(
            mode=mode,
            reduction=reduction,
            from_logits=from_logits,
            activation=activation,
            ignore_index=ignore_index,
            threshold=threshold,
            num_classes=num_classes,
            class_weights=class_weights,
            zero_division=zero_division,
        )

    def forward(
        self,
        output: Union[torch.LongTensor, torch.FloatTensor],
        target: torch.LongTensor,
    ):
        tp, fp, fn, tn = self._get_stats(
            output,
            target
        )
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
       from_logits (bool, optional): If True, the input is expected to be the raw output of a model. Default is True.
       activation (torch.nn.Module, optional): An activation function to apply to the output of the model. Default is None.
       ignore_index (int, optional): Specifies a target value that is ignored and does not contribute to the metric calculation. Default is None.
       threshold (float, optional): Threshold value for binarizing the output. Default is None.
       num_classes (int, optional): Number of classes for the metric calculation. Default is None.
       class_weights (torch.Tensor, optional): A manual rescaling weight given to each class. Default is None.
       zero_division (float): Value to return when there is a zero division. Default is 1.0.
    """
    def __init__(
        self,
        mode: str = "binary",
        reduction=None,
        from_logits=True,
        activation=None,
        ignore_index=None,
        threshold=None,
        num_classes=None,
        class_weights=None,
        zero_division=1.0,
    ):
        super(Recall, self).__init__(
            mode=mode,
            reduction=reduction,
            from_logits=from_logits,
            activation=activation,
            ignore_index=ignore_index,
            threshold=threshold,
            num_classes=num_classes,
            class_weights=class_weights,
            zero_division=zero_division,
        )

    def forward(
        self,
        output: Union[torch.LongTensor, torch.FloatTensor],
        target: torch.LongTensor,
    ):
        tp, fp, fn, tn = self._get_stats(
            output,
            target)
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
       from_logits (bool, optional): If True, the input is expected to be the raw output of a model. Default is True.
       activation (torch.nn.Module, optional): An activation function to apply to the output of the model. Default is None.
       ignore_index (int, optional): Specifies a target value that is ignored and does not contribute to the metric calculation. Default is None.
       threshold (float, optional): Threshold value for binarizing the output. Default is None.
       num_classes (int, optional): Number of classes for the metric calculation. Default is None.
       class_weights (torch.Tensor, optional): A manual rescaling weight given to each class. Default is None.
       zero_division (float): Value to return when there is a zero division. Default is 1.0.
    """
    def __init__(
        self,
        mode: str = "binary",
        reduction=None,
        from_logits=True,
        activation=None,
        ignore_index=None,
        threshold=None,
        num_classes=None,
        class_weights=None,
        zero_division=1.0,
    ):
        super(F1Score, self).__init__(
            mode=mode,
            reduction=reduction,
            from_logits=from_logits,
            activation=activation,
            ignore_index=ignore_index,
            threshold=threshold,
            num_classes=num_classes,
            class_weights=class_weights,
            zero_division=zero_division,
        )

    def forward(
        self,
        output: Union[torch.LongTensor, torch.FloatTensor],
        target: torch.LongTensor,
    ):
        tp, fp, fn, tn = self._get_stats(
            output,
            target)
        return f1_score(
            tp=tp,
            fp=fp,
            fn=fn,
            tn=tn,
            reduction=self.reduction,
            class_weights=self.class_weights,
            zero_division=self.zero_division,
        )
