import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset

from deepml.utils import (
    blend,
    get_random_samples_batch_from_dataset,
    get_random_samples_batch_from_loader,
    transform_input,
    transform_target,
)


class TestTransformTarget(unittest.TestCase):

    def test_scalar_tensor_no_classes(self):
        target = torch.tensor(3.14159)
        result = transform_target(target)
        self.assertAlmostEqual(result, 3.14, delta=1e-2)

    def test_1d_single_element_tensor(self):
        # ndim==1, shape[0]==1 -> .item() converts to float, then returned as-is (no classes)
        target = torch.tensor([2.5])
        result = transform_target(target)
        # After .item(), it's a Python float — returned as-is since no further Tensor checks apply
        self.assertAlmostEqual(result, 2.5, delta=1e-4)

    def test_tensor_with_classes_single(self):
        # A 0-d scalar tensor with no classes returns rounded value
        target = torch.tensor(2.718)
        result = transform_target(target)
        self.assertAlmostEqual(result, 2.72, delta=0.01)

    def test_tensor_multilabel(self):
        classes = ["cat", "dog", "bird"]
        target = torch.tensor([1, 0, 1])
        result = transform_target(target, classes=classes)
        self.assertEqual(result, "cat,bird")

    def test_int_with_classes(self):
        classes = ["apple", "banana", "cherry"]
        result = transform_target(1, classes=classes)
        self.assertEqual(result, "banana")

    def test_int_without_classes(self):
        result = transform_target(42)
        self.assertEqual(result, 42)


class TestTransformInput(unittest.TestCase):

    def test_permutes_bchw_to_bhwc(self):
        x = torch.randn(2, 3, 8, 8)
        result = transform_input(x)
        self.assertEqual(result.shape, (2, 8, 8, 3))

    def test_with_inverse_transform(self):
        x = torch.randn(2, 3, 8, 8)
        inverse = lambda t: t * 2
        result = transform_input(x, image_inverse_transform=inverse)
        self.assertEqual(result.shape, (2, 8, 8, 3))


class TestGetRandomSamplesBatchFromDataset(unittest.TestCase):

    def _make_dataset(self, size=20):
        x = torch.randn(size, 3, 8, 8)
        y = torch.randint(0, 2, (size,))
        return TensorDataset(x, y)

    def test_returns_correct_number_of_samples(self):
        dataset = self._make_dataset()
        batch = get_random_samples_batch_from_dataset(dataset, samples=5)
        self.assertEqual(len(batch), 5)

    def test_default_samples(self):
        dataset = self._make_dataset(size=20)
        batch = get_random_samples_batch_from_dataset(dataset)
        self.assertEqual(len(batch), 8)

    def test_empty_dataset_raises(self):
        dataset = TensorDataset(torch.empty(0, 3, 8, 8), torch.empty(0))
        with self.assertRaises(ValueError):
            get_random_samples_batch_from_dataset(dataset)


class TestGetRandomSamplesBatchFromLoader(unittest.TestCase):

    def _make_loader(self, size=20, batch_size=4):
        x = torch.randn(size, 3, 8, 8)
        y = torch.randint(0, 2, (size,))
        dataset = TensorDataset(x, y)
        return DataLoader(dataset, batch_size=batch_size)

    def test_returns_batch(self):
        loader = self._make_loader()
        batch = get_random_samples_batch_from_loader(loader)
        # TensorDataset returns tuples; collate yields tuple of tensors
        self.assertEqual(len(batch), 2)

    def test_custom_samples(self):
        loader = self._make_loader()
        batch = get_random_samples_batch_from_loader(loader, samples=3)
        self.assertEqual(batch[0].shape[0], 3)


class TestBlend(unittest.TestCase):

    def test_grayscale_3d_mask(self):
        image = torch.randint(0, 200, (2, 1, 8, 8), dtype=torch.float32)
        mask = torch.randint(0, 200, (2, 8, 8), dtype=torch.float32)
        result = blend(image, mask)
        self.assertEqual(result.shape, (2, 1, 8, 8))
        self.assertEqual(result.dtype, torch.uint8)

    def test_rgb_4d_mask(self):
        image = torch.randint(0, 200, (2, 3, 8, 8), dtype=torch.float32)
        mask = torch.randint(0, 200, (2, 1, 8, 8), dtype=torch.float32)
        result = blend(image, mask)
        self.assertEqual(result.shape, (2, 3, 8, 8))

    def test_rgb_3d_mask(self):
        image = torch.randint(0, 200, (2, 3, 8, 8), dtype=torch.float32)
        mask = torch.randint(0, 200, (2, 8, 8), dtype=torch.float32)
        result = blend(image, mask)
        self.assertEqual(result.shape, (2, 3, 8, 8))

    def test_non_4d_image_raises(self):
        image = torch.randn(2, 8, 8)  # 3D
        mask = torch.randn(2, 8, 8)
        with self.assertRaises(AssertionError):
            blend(image, mask)

    def test_values_clipped_to_uint8(self):
        image = torch.full((1, 1, 4, 4), 200.0)
        mask = torch.full((1, 4, 4), 200.0)
        result = blend(image, mask, alpha=1.0, beta=1.0)
        self.assertEqual(result.max().item(), 255)


if __name__ == "__main__":
    unittest.main()
