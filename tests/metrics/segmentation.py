import pytest
import torch

from deepml.metrics.segmentation import IoUScore, Precision, Recall


def test_precision_binary_classification():
    precision_metric = Precision(mode="binary", threshold=0.5)
    output = torch.tensor([[[[0.8, -0.01], [-0.5, 0.6]]]])
    target = torch.tensor([[[[1, 0], [0, 1]]]])
    result = precision_metric(output, target)
    assert result.item() == 1.0


def test_precision_recall_binary_custom_activation():
    gt = torch.tensor([[[[0, 1, 1], [0, 0, 0], [1, 1, 0]]]])
    pred = torch.tensor([[[[1, 0, 1], [1, 0, 0], [1, 1, 0]]]])

    precision = Precision(mode="binary", activation=lambda a: a, threshold=0.5)
    recall = Recall(mode="binary", activation=lambda a: a, threshold=0.5)

    assert pytest.approx(precision(pred, gt), 0.001) == 0.6
    assert pytest.approx(recall(pred, gt), 0.001) == 0.75


def test_precision_recall_multiclass_micro():
    gt = torch.tensor([[[0, 1, 2], [0, 1, 0], [2, 1, 0]]])
    # pred_class = torch.tensor([[[0, 1, 0], [1, 2, 0], [2, 1, 1]]])

    class1_prob = torch.tensor([[0.7, 0.2, 0.7], [0.2, 0.1, 0.7], [0.1, 0.2, 0.2]])
    class2_prob = torch.tensor([[0.2, 0.7, 0.2], [0.7, 0.2, 0.2], [0.2, 0.7, 0.7]])
    class3_prob = torch.tensor([[0.1, 0.1, 0.1], [0.1, 0.7, 0.1], [0.7, 0.1, 0.1]])
    probs = torch.stack([class1_prob, class2_prob, class3_prob]).unsqueeze(dim=0)

    precision = Precision(mode="multiclass", reduction="micro", num_classes=3)
    recall = Recall(mode="multiclass", reduction="micro", num_classes=3)

    assert pytest.approx(precision(probs, gt), 0.01) == 0.5556
    assert pytest.approx(recall(probs, gt), 0.01) == 0.5556


def test_precision_recall_multiclass_without_reduction():
    gt = torch.tensor([[[0, 1, 2], [0, 1, 0], [2, 1, 0]]])
    # pred_class = torch.tensor([[[0, 1, 0], [1, 2, 0], [2, 1, 1]]])

    class1_prob = torch.tensor([[0.7, 0.2, 0.7], [0.2, 0.1, 0.7], [0.1, 0.2, 0.2]])
    class2_prob = torch.tensor([[0.2, 0.7, 0.2], [0.7, 0.2, 0.2], [0.2, 0.7, 0.7]])
    class3_prob = torch.tensor([[0.1, 0.1, 0.1], [0.1, 0.7, 0.1], [0.7, 0.1, 0.1]])
    probs = torch.stack([class1_prob, class2_prob, class3_prob]).unsqueeze(dim=0)

    precision = Precision(mode="multiclass", num_classes=3)
    recall = Recall(mode="multiclass", num_classes=3)

    assert pytest.approx(precision(probs, gt), 0.01) == torch.tensor(
        [[0.6667, 0.5000, 0.5000]]
    )
    assert pytest.approx(recall(probs, gt), 0.01) == torch.tensor(
        [[0.5000, 0.6667, 0.5000]]
    )


def test_precision_recall_multiclass_macro_reduction():
    gt = torch.tensor([[[0, 1, 2], [0, 1, 0], [2, 1, 0]]])
    # pred_class = torch.tensor([[[0, 1, 0], [1, 2, 0], [2, 1, 1]]])

    class1_prob = torch.tensor([[0.7, 0.2, 0.7], [0.2, 0.1, 0.7], [0.1, 0.2, 0.2]])
    class2_prob = torch.tensor([[0.2, 0.7, 0.2], [0.7, 0.2, 0.2], [0.2, 0.7, 0.7]])
    class3_prob = torch.tensor([[0.1, 0.1, 0.1], [0.1, 0.7, 0.1], [0.7, 0.1, 0.1]])
    probs = torch.stack([class1_prob, class2_prob, class3_prob]).unsqueeze(dim=0)

    precision = Precision(
        mode="multiclass", num_classes=3, reduction="macro", class_weights=[1, 0, 0]
    )
    recall = Recall(
        mode="multiclass", num_classes=3, reduction="macro", class_weights=[1, 0, 0]
    )

    # precision and recall of class 1 is divided by number of classes
    # 0.67 / 3 = 0.223 and 0.5/3 = 0.166

    assert pytest.approx(precision(probs, gt), 0.01) == 0.223
    assert pytest.approx(recall(probs, gt), 0.01) == 0.166


def test_precision_recall_target_class_index():
    gt = torch.tensor([[[0, 1, 2], [0, 1, 0], [2, 1, 0]]])
    class1_prob = torch.tensor([[0.7, 0.2, 0.7], [0.2, 0.1, 0.7], [0.1, 0.2, 0.2]])
    class2_prob = torch.tensor([[0.2, 0.7, 0.2], [0.7, 0.2, 0.2], [0.2, 0.7, 0.7]])
    class3_prob = torch.tensor([[0.1, 0.1, 0.1], [0.1, 0.7, 0.1], [0.7, 0.1, 0.1]])

    probs = torch.stack([class1_prob, class2_prob, class3_prob]).unsqueeze(dim=0)
    precision = Precision(mode="multiclass", num_classes=3, target_class_index=0)
    assert pytest.approx(precision(probs, gt), 0.001) == 0.667

    precision = Precision(mode="multiclass", num_classes=3, target_class_index=1)
    assert pytest.approx(precision(probs, gt), 0.001) == 0.5

    recall = Recall(mode="multiclass", num_classes=3, target_class_index=1)
    assert pytest.approx(recall(probs, gt), 0.001) == 0.667

    recall = Recall(mode="multiclass", num_classes=3, target_class_index=2)
    assert pytest.approx(recall(probs, gt), 0.001) == 0.5

    # Test for multiple images
    probs = torch.stack([class1_prob, class2_prob, class3_prob]).unsqueeze(dim=0)
    precision = Precision(
        mode="multiclass", num_classes=3, target_class_index=0, reduction="macro"
    )

    print(
        precision(
            torch.concatenate([probs, probs], dim=0), torch.concatenate([gt, gt], dim=0)
        )
    )


def test_precision_recall_multilabel():
    gt_class_0 = torch.tensor([[1, 0, 1], [0, 1, 1], [0, 0, 0]])

    gt_class_1 = torch.tensor([[1, 0, 0], [1, 1, 0], [1, 0, 0]])
    gt_class_2 = torch.tensor([[1, 1, 0], [0, 1, 1], [0, 0, 0]])
    gt = torch.stack([gt_class_0, gt_class_1, gt_class_2], dim=0).unsqueeze(dim=0)

    pred_class_0 = torch.tensor([[0, 1, 1], [1, 0, 1], [0, 1, 0]])

    pred_class_1 = torch.tensor([[1, 0, 0], [1, 1, 1], [0, 0, 0]])

    pred_class_2 = torch.tensor([[1, 0, 1], [0, 0, 0], [1, 1, 1]])

    pred = torch.stack([pred_class_0, pred_class_1, pred_class_2], dim=0).unsqueeze(
        dim=0
    )

    precision = Precision(mode="multilabel", num_classes=3, threshold=0.6)
    recall = Recall(mode="multilabel", num_classes=3, threshold=0.6)

    assert pytest.approx(precision(pred, gt), 0.001) == torch.tensor(
        [[0.40, 0.75, 0.20]]
    )
    assert pytest.approx(recall(pred, gt), 0.001) == torch.tensor([[0.5, 0.75, 0.25]])

    precision = Precision(
        mode="multilabel", num_classes=3, threshold=0.6, target_class_index=2
    )
    recall = Recall(
        mode="multilabel", num_classes=3, threshold=0.6, target_class_index=2
    )
    assert pytest.approx(precision(pred, gt), 0.001) == 0.20
    assert pytest.approx(recall(pred, gt), 0.001) == 0.25


def test_jaccard_index_binary_custom_activation():
    gt = torch.tensor([[[[0, 1, 1], [0, 0, 1], [1, 1, 1]]]])
    pred = torch.tensor([[[[1, 1, 1], [1, 0, 0], [1, 1, 0]]]])

    iou = IoUScore(mode="binary", activation=lambda a: a, threshold=0.5)

    assert pytest.approx(iou(pred, gt), 0.001) == 0.5

    gt = torch.tensor([[[[1, 1, 0], [1, 1, 0], [1, 1, 0]]]])
    pred = torch.tensor([[[[1, 1, 1], [1, 1, 1], [1, 1, 1]]]])

    assert pytest.approx(iou(pred, gt), 0.001) == 0.667
