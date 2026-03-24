import unittest

import torch

from deepml.metrics import classification, commons, segmentation


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
        self.assertAlmostEqual(acc(output, target), 0.5833, delta=1e-4)

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
