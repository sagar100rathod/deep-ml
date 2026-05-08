import unittest

import torch

from deepml.losses import (
    AngularPenaltySMLoss,
    ContrastiveLoss,
    JaccardLoss,
    RMSELoss,
    WeightedBCEWithLogitsLoss,
)


class TestJaccardLoss(unittest.TestCase):

    def test_binary_perfect_prediction(self):
        loss_fn = JaccardLoss(is_multiclass=False)
        output = torch.ones(2, 1, 4, 4) * 10  # high logits -> sigmoid ~ 1
        target = torch.ones(2, 1, 4, 4)
        loss = loss_fn(output, target)
        self.assertAlmostEqual(loss.item(), 0.0, delta=0.01)

    def test_binary_worst_prediction(self):
        loss_fn = JaccardLoss(is_multiclass=False)
        output = torch.ones(2, 1, 4, 4) * -10  # sigmoid ~ 0
        target = torch.ones(2, 1, 4, 4)
        loss = loss_fn(output, target)
        self.assertAlmostEqual(loss.item(), 1.0, delta=0.01)

    def test_multiclass(self):
        loss_fn = JaccardLoss(is_multiclass=True)
        output = torch.randn(2, 3, 4, 4)
        target = torch.rand(2, 3, 4, 4)
        loss = loss_fn(output, target)
        self.assertTrue(0.0 <= loss.item() <= 1.0)

    def test_returns_scalar(self):
        loss_fn = JaccardLoss(is_multiclass=False)
        output = torch.randn(2, 1, 8, 8)
        target = torch.rand(2, 1, 8, 8)
        loss = loss_fn(output, target)
        self.assertEqual(loss.ndim, 0)


class TestRMSELoss(unittest.TestCase):

    def test_perfect_prediction(self):
        loss_fn = RMSELoss()
        x = torch.tensor([1.0, 2.0, 3.0])
        loss = loss_fn(x, x)
        self.assertAlmostEqual(loss.item(), 1e-3, delta=1e-2)

    def test_known_value(self):
        loss_fn = RMSELoss(eps=0.0)
        output = torch.tensor([0.0, 0.0])
        target = torch.tensor([3.0, 4.0])
        # MSE = (9+16)/2 = 12.5, RMSE = sqrt(12.5)
        expected = 12.5**0.5
        loss = loss_fn(output, target)
        self.assertAlmostEqual(loss.item(), expected, delta=1e-4)

    def test_non_negative(self):
        loss_fn = RMSELoss()
        output = torch.randn(10)
        target = torch.randn(10)
        self.assertGreaterEqual(loss_fn(output, target).item(), 0.0)


class TestWeightedBCEWithLogitsLoss(unittest.TestCase):

    def test_forward_runs(self):
        loss_fn = WeightedBCEWithLogitsLoss(w_p=1.0, w_n=1.0)
        logits = torch.tensor([2.0, -1.0, 0.5, -0.5])
        labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
        loss = loss_fn(logits, labels)
        self.assertEqual(loss.ndim, 0)
        self.assertGreater(loss.item(), 0.0)

    def test_with_different_weights(self):
        loss_fn_balanced = WeightedBCEWithLogitsLoss(w_p=1.0, w_n=1.0)
        loss_fn_pos_heavy = WeightedBCEWithLogitsLoss(w_p=2.0, w_n=1.0)
        logits = torch.tensor([2.0, -1.0, 0.5, -0.5])
        labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
        # higher weight on positives should give different loss
        self.assertNotAlmostEqual(
            loss_fn_balanced(logits, labels).item(),
            loss_fn_pos_heavy(logits, labels).item(),
            places=4,
        )


class TestContrastiveLoss(unittest.TestCase):

    def test_similar_pairs_zero_distance(self):
        loss_fn = ContrastiveLoss(margin=2.0)
        emb = torch.zeros(4, 8)
        label = torch.ones(4)  # all similar
        loss = loss_fn((emb, emb), label)
        self.assertAlmostEqual(loss.item(), 0.0, delta=1e-5)

    def test_dissimilar_pairs_beyond_margin(self):
        loss_fn = ContrastiveLoss(margin=1.0)
        emb1 = torch.zeros(4, 8)
        emb2 = torch.ones(4, 8) * 100  # very far apart
        label = torch.zeros(4)  # all dissimilar
        loss = loss_fn((emb1, emb2), label)
        # clamp ensures no penalty when distance >> margin
        self.assertAlmostEqual(loss.item(), 0.0, delta=1e-4)

    def test_with_label_transform(self):
        loss_fn = ContrastiveLoss(margin=2.0, label_transform=lambda l: l[:, 0])
        emb1 = torch.randn(4, 8)
        emb2 = torch.randn(4, 8)
        labels = torch.ones(4, 2)
        loss = loss_fn((emb1, emb2), labels)
        self.assertEqual(loss.ndim, 0)

    def test_returns_scalar(self):
        loss_fn = ContrastiveLoss()
        emb1 = torch.randn(8, 16)
        emb2 = torch.randn(8, 16)
        label = torch.randint(0, 2, (8,)).float()
        loss = loss_fn((emb1, emb2), label)
        self.assertEqual(loss.ndim, 0)


class TestAngularPenaltySMLoss(unittest.TestCase):

    def _make_inputs(self, batch=8, in_feat=64, num_classes=10):
        x = torch.randn(batch, in_feat)
        labels = torch.randint(0, num_classes, (batch,))
        return x, labels

    def test_arcface_forward(self):
        loss_fn = AngularPenaltySMLoss(
            in_features=64, out_features=10, loss_type="arcface"
        )
        x, labels = self._make_inputs()
        loss = loss_fn(x, labels)
        self.assertEqual(loss.ndim, 0)

    def test_cosface_forward(self):
        loss_fn = AngularPenaltySMLoss(
            in_features=64, out_features=10, loss_type="cosface"
        )
        x, labels = self._make_inputs()
        loss = loss_fn(x, labels)
        self.assertEqual(loss.ndim, 0)

    def test_sphereface_forward(self):
        loss_fn = AngularPenaltySMLoss(
            in_features=64, out_features=10, loss_type="sphereface"
        )
        x, labels = self._make_inputs()
        loss = loss_fn(x, labels)
        self.assertEqual(loss.ndim, 0)

    def test_invalid_loss_type(self):
        with self.assertRaises(AssertionError):
            AngularPenaltySMLoss(in_features=64, out_features=10, loss_type="invalid")

    def test_custom_s_and_m(self):
        loss_fn = AngularPenaltySMLoss(
            in_features=32, out_features=5, loss_type="arcface", s=16.0, m=0.3
        )
        self.assertEqual(loss_fn.s, 16.0)
        self.assertEqual(loss_fn.m, 0.3)

    def test_default_hyperparams_arcface(self):
        loss_fn = AngularPenaltySMLoss(
            in_features=32, out_features=5, loss_type="arcface"
        )
        self.assertEqual(loss_fn.s, 64.0)
        self.assertEqual(loss_fn.m, 0.5)

    def test_default_hyperparams_cosface(self):
        loss_fn = AngularPenaltySMLoss(
            in_features=32, out_features=5, loss_type="cosface"
        )
        self.assertEqual(loss_fn.s, 30.0)
        self.assertEqual(loss_fn.m, 0.4)


if __name__ == "__main__":
    unittest.main()
