import unittest

import pytest
import torch

from deepml.metrics.segmentation import Accuracy, F1Score, IoUScore, Precision, Recall


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

    precision = Precision(mode="multiclass", num_classes=3, reduction=None)
    recall = Recall(mode="multiclass", num_classes=3, reduction=None)

    assert precision(probs, gt) == pytest.approx(
        torch.tensor([0.6667, 0.5000, 0.5000]), abs=0.01
    )
    assert recall(probs, gt) == pytest.approx(
        torch.tensor([0.5000, 0.6667, 0.5000]), abs=0.01
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

    precision = Precision(
        mode="multilabel", num_classes=3, threshold=0.6, reduction=None
    )
    recall = Recall(mode="multilabel", num_classes=3, threshold=0.6, reduction=None)

    assert precision(pred, gt) == pytest.approx(
        torch.tensor([0.40, 0.75, 0.20]), abs=0.001
    )
    assert recall(pred, gt) == pytest.approx(torch.tensor([0.5, 0.75, 0.25]), abs=0.001)

    precision = Precision(
        mode="multilabel",
        num_classes=3,
        threshold=0.6,
        target_class_index=2,
        reduction=None,
    )
    recall = Recall(
        mode="multilabel",
        num_classes=3,
        threshold=0.6,
        target_class_index=2,
        reduction=None,
    )
    assert pytest.approx(precision(pred, gt), 0.001) == 0.20
    assert pytest.approx(recall(pred, gt), 0.001) == 0.25


def test_jaccard_index_binary_custom_activation():
    gt = torch.tensor([[[[0, 1, 1], [0, 0, 1], [1, 1, 1]]]])
    pred = torch.tensor([[[[1, 1, 1], [1, 0, 0], [1, 1, 0]]]])

    iou = IoUScore(mode="binary", activation=lambda a: a, threshold=0.5)

    assert pytest.approx(iou(pred, gt), 0.001) == 0.5

    iou.reset()  # reset accumulated state before testing on new data
    gt = torch.tensor([[[[1, 1, 0], [1, 1, 0], [1, 1, 0]]]])
    pred = torch.tensor([[[[1, 1, 1], [1, 1, 1], [1, 1, 1]]]])

    assert pytest.approx(iou(pred, gt), 0.001) == 0.667


def test_accuracy_binary():
    gt = torch.tensor([[[[0, 1, 1], [0, 0, 0], [1, 1, 0]]]])
    pred = torch.tensor([[[[1, 0, 1], [1, 0, 0], [1, 1, 0]]]])

    acc = Accuracy(mode="binary", activation=lambda a: a, threshold=0.5)

    # Correct predictions: 6 out of 9 pixels
    assert pytest.approx(acc(pred, gt).item(), 0.001) == 0.6667


def test_accuracy_binary_all_correct():
    gt = torch.tensor([[[[1, 0], [0, 1]]]])
    pred = torch.tensor([[[[0.8, -0.01], [-0.5, 0.6]]]])

    acc = Accuracy(mode="binary", threshold=0.5)
    assert acc(pred, gt).item() == 1.0


def test_accuracy_multiclass_micro():
    gt = torch.tensor([[[0, 1, 2], [0, 1, 0], [2, 1, 0]]])

    class1_prob = torch.tensor([[0.7, 0.2, 0.7], [0.2, 0.1, 0.7], [0.1, 0.2, 0.2]])
    class2_prob = torch.tensor([[0.2, 0.7, 0.2], [0.7, 0.2, 0.2], [0.2, 0.7, 0.7]])
    class3_prob = torch.tensor([[0.1, 0.1, 0.1], [0.1, 0.7, 0.1], [0.7, 0.1, 0.1]])
    probs = torch.stack([class1_prob, class2_prob, class3_prob]).unsqueeze(dim=0)

    acc = Accuracy(mode="multiclass", reduction="micro", num_classes=3)
    # torchmetrics computes simple pixel accuracy (correct/total = 5/9)
    assert pytest.approx(acc(probs, gt).item(), 0.001) == 0.5556


def test_accuracy_multiclass_target_class_index():
    gt = torch.tensor([[[0, 1, 2], [0, 1, 0], [2, 1, 0]]])

    class1_prob = torch.tensor([[0.7, 0.2, 0.7], [0.2, 0.1, 0.7], [0.1, 0.2, 0.2]])
    class2_prob = torch.tensor([[0.2, 0.7, 0.2], [0.7, 0.2, 0.2], [0.2, 0.7, 0.7]])
    class3_prob = torch.tensor([[0.1, 0.1, 0.1], [0.1, 0.7, 0.1], [0.7, 0.1, 0.1]])
    probs = torch.stack([class1_prob, class2_prob, class3_prob]).unsqueeze(dim=0)

    acc = Accuracy(mode="multiclass", num_classes=3, target_class_index=0)
    result = acc(probs, gt)
    assert result.item() > 0.0


def test_accuracy_multilabel():
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

    acc = Accuracy(mode="multilabel", num_classes=3, threshold=0.6)
    result = acc(pred, gt)
    assert result is not None
    assert pytest.approx(result.item(), 0.001) == 0.4815


def test_f1score_binary():
    gt = torch.tensor([[[[0, 1, 1], [0, 0, 0], [1, 1, 0]]]])
    pred = torch.tensor([[[[1, 0, 1], [1, 0, 0], [1, 1, 0]]]])

    f1 = F1Score(mode="binary", activation=lambda a: a, threshold=0.5)
    assert pytest.approx(f1(pred, gt).item(), 0.001) == 0.6667


def test_f1score_multiclass_micro():
    gt = torch.tensor([[[0, 1, 2], [0, 1, 0], [2, 1, 0]]])

    class1_prob = torch.tensor([[0.7, 0.2, 0.7], [0.2, 0.1, 0.7], [0.1, 0.2, 0.2]])
    class2_prob = torch.tensor([[0.2, 0.7, 0.2], [0.7, 0.2, 0.2], [0.2, 0.7, 0.7]])
    class3_prob = torch.tensor([[0.1, 0.1, 0.1], [0.1, 0.7, 0.1], [0.7, 0.1, 0.1]])
    probs = torch.stack([class1_prob, class2_prob, class3_prob]).unsqueeze(dim=0)

    f1 = F1Score(mode="multiclass", reduction="micro", num_classes=3)
    assert pytest.approx(f1(probs, gt), 0.01) == 0.5556


def _make_binary_batch(batch=2, h=8, w=8, logit_val=5.0):
    """Returns (output, target) for binary segmentation where predictions are perfect."""
    target = torch.randint(0, 2, (batch, 1, h, w)).long()
    output = target.float() * logit_val + (1 - target.float()) * (-logit_val)
    return output, target


def _make_multiclass_batch(batch=2, num_classes=3, h=8, w=8):
    target = torch.randint(0, num_classes, (batch, h, w)).long()
    output = torch.zeros(batch, num_classes, h, w)
    for b in range(batch):
        for c in range(num_classes):
            output[b, c] = (target[b] == c).float() * 10.0
    return output, target


def _make_binary_metric_kwargs():
    """Return kwargs for binary segmentation metrics with explicit threshold."""
    return {"mode": "binary", "threshold": 0.5}


class TestSegmentationMetricValidation(unittest.TestCase):

    def test_invalid_mode_raises(self):
        with self.assertRaises(ValueError):
            Precision(mode="bad_mode")

    def test_binary_ignore_index_raises(self):
        with self.assertRaises(ValueError):
            Precision(mode="binary", ignore_index=0)

    def test_multiclass_requires_num_classes(self):
        with self.assertRaises(ValueError):
            Precision(mode="multiclass")

    def test_target_class_index_exceeds_num_classes(self):
        with self.assertRaises(ValueError):
            Precision(mode="multiclass", num_classes=3, target_class_index=5)

    def test_binary_target_class_index_raises(self):
        with self.assertRaises(ValueError):
            Precision(mode="binary", target_class_index=0)


class TestBinaryPrecision(unittest.TestCase):

    def test_perfect_precision(self):
        metric = Precision(mode="binary", threshold=0.5)
        output, target = _make_binary_batch()
        result = metric(output, target).item()
        self.assertAlmostEqual(result, 1.0, delta=1e-4)

    def test_returns_scalar(self):
        metric = Precision(mode="binary", threshold=0.5)
        output, target = _make_binary_batch()
        result = metric(output, target)
        self.assertEqual(result.ndim, 0)


class TestBinaryRecall(unittest.TestCase):

    def test_perfect_recall(self):
        metric = Recall(mode="binary", threshold=0.5)
        output, target = _make_binary_batch()
        result = metric(output, target).item()
        self.assertAlmostEqual(result, 1.0, delta=1e-4)


class TestBinaryF1Score(unittest.TestCase):

    def test_perfect_f1(self):
        metric = F1Score(mode="binary", threshold=0.5)
        output, target = _make_binary_batch()
        result = metric(output, target).item()
        self.assertAlmostEqual(result, 1.0, delta=1e-4)


class TestBinaryAccuracy(unittest.TestCase):

    def test_perfect_accuracy(self):
        metric = Accuracy(mode="binary", threshold=0.5)
        output, target = _make_binary_batch()
        result = metric(output, target).item()
        self.assertAlmostEqual(result, 1.0, delta=1e-4)


class TestBinaryIoUScore(unittest.TestCase):

    def test_perfect_iou(self):
        metric = IoUScore(mode="binary", threshold=0.5)
        output, target = _make_binary_batch()
        result = metric(output, target).item()
        self.assertAlmostEqual(result, 1.0, delta=1e-4)


class TestMulticlassMetrics(unittest.TestCase):

    def test_multiclass_precision(self):
        metric = Precision(mode="multiclass", num_classes=3)
        output, target = _make_multiclass_batch(num_classes=3)
        result = metric(output, target)
        self.assertGreaterEqual(result.item(), 0.0)

    def test_multiclass_recall(self):
        metric = Recall(mode="multiclass", num_classes=3)
        output, target = _make_multiclass_batch(num_classes=3)
        result = metric(output, target)
        self.assertGreaterEqual(result.item(), 0.0)

    def test_multiclass_f1(self):
        metric = F1Score(mode="multiclass", num_classes=3)
        output, target = _make_multiclass_batch(num_classes=3)
        result = metric(output, target)
        self.assertGreaterEqual(result.item(), 0.0)

    def test_multiclass_iou(self):
        metric = IoUScore(mode="multiclass", num_classes=3)
        output, target = _make_multiclass_batch(num_classes=3)
        result = metric(output, target)
        self.assertGreaterEqual(result.item(), 0.0)

    def test_target_class_index(self):
        metric = Precision(mode="multiclass", num_classes=3, target_class_index=1)
        output, target = _make_multiclass_batch(num_classes=3)
        result = metric(output, target)
        self.assertEqual(result.ndim, 0)

    def test_with_ignore_index(self):
        metric = Precision(mode="multiclass", num_classes=3, ignore_index=0)
        output, target = _make_multiclass_batch(num_classes=3)
        result = metric(output, target)
        self.assertGreaterEqual(result.item(), 0.0)

    def test_with_callable(self):
        def noop(output, target):
            return output, target

        metric = Precision(mode="binary", threshold=0.5, callable=noop)
        output, target = _make_binary_batch()
        result = metric(output, target)
        self.assertAlmostEqual(result.item(), 1.0, delta=1e-4)


class TestStatefulAccumulation(unittest.TestCase):

    def test_accumulated_iou_differs_from_sma(self):
        """Global IoU from accumulated counts must differ from mean of per-batch IoUs."""
        iou = IoUScore(mode="binary", threshold=0.5)

        # batch 1: small, mostly correct
        gt1 = torch.tensor([[[[1, 1], [1, 1]]]])
        pred1 = torch.tensor([[[[5.0, 5.0], [5.0, 5.0]]]])
        val1 = iou(pred1, gt1).item()

        # batch 2: larger, mostly wrong
        gt2 = torch.tensor([[[[0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1]]]])
        pred2 = torch.tensor(
            [
                [
                    [
                        [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
                        [-5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0],
                    ]
                ]
            ]
        )
        val_accumulated = iou(pred2, gt2).item()

        sma_approx = (val1 + val_accumulated) / 2
        # accumulated value should differ from simple average of per-batch values
        self.assertNotAlmostEqual(val_accumulated, sma_approx, places=3)

    def test_reset_clears_state(self):
        """After reset(), compute() on one batch equals single-batch IoU exactly."""
        iou = IoUScore(mode="binary", threshold=0.5)
        gt = torch.tensor([[[[1, 0], [0, 1]]]])
        pred = torch.tensor([[[[5.0, -5.0], [-5.0, 5.0]]]])

        iou(pred, gt)  # first call accumulates
        iou.reset()
        result = iou(pred, gt)  # fresh accumulation — equals single-batch IoU

        iou_fresh = IoUScore(mode="binary", threshold=0.5)
        expected = iou_fresh(pred, gt)
        self.assertAlmostEqual(result.item(), expected.item(), places=5)

    def test_is_stateful_flag(self):
        self.assertTrue(IoUScore.is_stateful)
        self.assertTrue(F1Score.is_stateful)
        self.assertTrue(Precision.is_stateful)
        self.assertTrue(Recall.is_stateful)
        self.assertTrue(Accuracy.is_stateful)
