import unittest

import torch

from deepml.metrics import classification, commons
from deepml.metrics.classification import MCC, Accuracy, FScore, Precision, Recall


class TestAccuracy(unittest.TestCase):

    def test_perfect_binary(self):
        acc = Accuracy()
        output = torch.tensor([5.0, -5.0, 5.0, -5.0])
        target = torch.tensor([1.0, 0.0, 1.0, 0.0])
        self.assertAlmostEqual(acc(output, target).item(), 1.0, delta=1e-4)

    def test_all_wrong_binary(self):
        acc = Accuracy()
        output = torch.tensor([-5.0, 5.0, -5.0, 5.0])
        target = torch.tensor([1.0, 0.0, 1.0, 0.0])
        self.assertAlmostEqual(acc(output, target).item(), 0.0, delta=1e-4)

    def test_multiclass_accuracy(self):
        acc = Accuracy(num_classes=2)
        output = torch.tensor([[5.0, 0.0], [0.0, 5.0]])
        target = torch.tensor([0.0, 1.0])
        self.assertAlmostEqual(acc(output, target).item(), 1.0, delta=1e-4)


class TestPrecision(unittest.TestCase):

    def test_binary_perfect(self):
        prec = Precision()
        output = torch.tensor([5.0, -5.0, 5.0])
        target = torch.tensor([1.0, 0.0, 1.0])
        self.assertAlmostEqual(prec(output, target).item(), 1.0, delta=1e-4)

    def test_binary_no_tp(self):
        prec = Precision()
        output = torch.tensor([-5.0, -5.0])
        target = torch.tensor([1.0, 1.0])
        result = prec(output, target).item()
        self.assertAlmostEqual(result, 0.0, delta=1e-3)

    def test_multiclass(self):
        prec = Precision(num_classes=3)
        output = torch.tensor([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]])
        target = torch.tensor([0.0, 1.0, 2.0])
        result = prec(output, target).item()
        self.assertGreater(result, 0.0)


class TestRecall(unittest.TestCase):

    def test_binary_perfect(self):
        rec = Recall()
        output = torch.tensor([5.0, -5.0, 5.0])
        target = torch.tensor([1.0, 0.0, 1.0])
        self.assertAlmostEqual(rec(output, target).item(), 1.0, delta=1e-4)

    def test_binary_missed_all(self):
        rec = Recall()
        output = torch.tensor([-5.0, -5.0])
        target = torch.tensor([1.0, 1.0])
        result = rec(output, target).item()
        self.assertAlmostEqual(result, 0.0, delta=1e-3)

    def test_multiclass(self):
        rec = Recall(num_classes=2)
        output = torch.tensor([[5.0, 0.0], [0.0, 5.0]])
        target = torch.tensor([0.0, 1.0])
        result = rec(output, target).item()
        self.assertGreater(result, 0.0)


class TestFScore(unittest.TestCase):

    def test_binary_perfect(self):
        fscore = FScore()
        output = torch.tensor([5.0, -5.0, 5.0])
        target = torch.tensor([1.0, 0.0, 1.0])
        self.assertAlmostEqual(fscore(output, target).item(), 1.0, delta=1e-4)

    def test_multiclass(self):
        fscore = FScore(num_classes=2)
        output = torch.tensor([[5.0, 0.0], [0.0, 5.0]])
        target = torch.tensor([0.0, 1.0])
        result = fscore(output, target).item()
        self.assertGreater(result, 0.0)

    def test_beta_param(self):
        fscore_b1 = FScore(beta=1.0)
        fscore_b2 = FScore(beta=2.0)
        output = torch.tensor([5.0, -5.0, 5.0, -5.0])
        target = torch.tensor([1.0, 0.0, 1.0, 0.0])
        r1 = fscore_b1(output, target).item()
        r2 = fscore_b2(output, target).item()
        self.assertGreater(r1, 0.0)
        self.assertGreater(r2, 0.0)
        self.assertLessEqual(r1, 1.0)
        self.assertLessEqual(r2, 1.0)


class TestMCC(unittest.TestCase):

    def test_binary_perfect(self):
        mcc = MCC()
        output = torch.tensor([5.0, -5.0, 5.0, -5.0])
        target = torch.tensor([1.0, 0.0, 1.0, 0.0])
        result = mcc(output, target).item()
        self.assertAlmostEqual(result, 1.0, delta=1e-4)

    def test_binary_all_wrong(self):
        mcc = MCC()
        output = torch.tensor([-5.0, 5.0, -5.0, 5.0])
        target = torch.tensor([1.0, 0.0, 1.0, 0.0])
        result = mcc(output, target).item()
        self.assertAlmostEqual(result, -1.0, delta=1e-3)

    def test_multiclass(self):
        mcc = MCC(num_classes=2)
        output = torch.tensor([[5.0, 0.0], [0.0, 5.0], [5.0, 0.0], [0.0, 5.0]])
        target = torch.tensor([0.0, 1.0, 0.0, 1.0])
        result = mcc(output, target).item()
        self.assertAlmostEqual(result, 1.0, delta=1e-4)


class TestIsStateful(unittest.TestCase):

    def test_all_classification_metrics_are_stateful(self):
        self.assertTrue(Accuracy.is_stateful)
        self.assertTrue(Precision.is_stateful)
        self.assertTrue(Recall.is_stateful)
        self.assertTrue(FScore.is_stateful)
        self.assertTrue(MCC.is_stateful)

    def test_reset_clears_state(self):
        prec = Precision()
        output = torch.tensor([5.0, -5.0])
        target = torch.tensor([1.0, 0.0])
        prec(output, target)
        prec.reset()
        result = prec(output, target)
        fresh = Precision()
        expected = fresh(output, target)
        self.assertAlmostEqual(result.item(), expected.item(), places=5)


class TestImageClassificationMetrics(unittest.TestCase):

    def test_binary_classification(self):
        target = torch.tensor([1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1], dtype=torch.int8)
        output = torch.tensor([1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0], dtype=torch.int8)

        self.assertEqual(commons.true_positives(output, target), 4)
        self.assertEqual(commons.false_positives(output, target), 3)
        self.assertEqual(commons.false_negatives(output, target), 3)
        self.assertEqual(commons.true_negatives(output, target), 2)

        output = torch.tensor(
            [0.6, 0.5, 0.3, 0.2, 0.8, 0.2, 0.1, 0.7, 0.49, 0.51, 0.8, 0.95]
        )
        acc = classification.Accuracy()
        # torchmetrics uses > threshold (same as old code) — 0.5 at threshold=0.5 → predicted negative
        self.assertAlmostEqual(acc(output, target).item(), 0.5833, delta=1e-3)

    def test_multiclass_classification(self):
        target = torch.tensor(
            [1, 4, 3, 2, 1, 1, 2, 3, 4, 2, 1, 2, 3, 4, 1, 2], dtype=torch.int8
        )
        output = torch.tensor(
            [1, 2, 3, 4, 1, 2, 2, 3, 4, 2, 3, 1, 3, 4, 2, 3], dtype=torch.int8
        )
        tp, fp, tn, fn = commons.multiclass_tp_fp_tn_fn(output, target)

        self.assertEqual(tp, 9)
        self.assertEqual(fp, 7)
        self.assertEqual(fn, 7)


if __name__ == "__main__":
    unittest.main()
