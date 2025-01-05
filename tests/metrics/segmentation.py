import pytest
import torch
from deepml.metrics.segmentation import Precision


def test_precision_binary_classification():
    precision_metric = Precision(mode="binary",
                                 threshold=0.5)
    output = torch.tensor([[[[0.8, -0.01], [-0.5, 0.6]]]])
    target = torch.tensor([[[[1, 0], [0, 1]]]])
    result = precision_metric(output, target)
    assert result.item() == 1.0

def test_precision_multiclass_classification():
    precision_metric = Precision(mode="multiclass",
                                 from_logits=False,
                                 activation=None,
                                 num_classes=3, reduction='micro')
    output = torch.tensor([[[0,  1, 2],
                          [1, 0, 2],
                          [2, 1, 0]]])
    target = torch.tensor([[[0, 1, 2],
                          [1, 0, 2],
                          [2, 1, 0]]])
    result = precision_metric(output, target)
    assert result.item() == 1

def test_precision_ignore_index():
    precision_metric = Precision(mode="multiclass",
                                 from_logits=False,
                                 num_classes=3,
                                 ignore_index=0,
                                 reduction='micro')
    output = torch.tensor([[[0,  1, 2],
                            [1, 0, 1],
                            [1, 2, 0]]])

    target = torch.tensor([[[0, 1, 2],
                            [1, 0, 1],
                            [0, 1, 0]]])

    result = precision_metric(output, target)
    assert pytest.approx(result.item(), 0.0001) == 0.8
