import unittest

import torch

from deepml import constants
from deepml.transforms import (
    DivideBy255,
    ImageInverseTransform,
    ImageNetInverseTransform,
    MulticlassSegmentationTargetTransform,
)


class TestImageInverseTransform(unittest.TestCase):

    def test_inverse_transform_restores_values(self):
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        transform = ImageInverseTransform(mean, std)
        # normalized image: (x - mean) / std => (1 - 0.5)/0.5 = 1
        img = torch.ones(3, 4, 4)  # CHW
        result = transform(img)
        # result = img * std + mean = 1 * 0.5 + 0.5 = 1.0
        self.assertAlmostEqual(result.mean().item(), 1.0, delta=1e-5)

    def test_output_shape_preserved(self):
        transform = ImageInverseTransform([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        img = torch.randn(3, 32, 32)
        result = transform(img)
        self.assertEqual(result.shape, img.shape)


class TestImageNetInverseTransform(unittest.TestCase):

    def test_uses_imagenet_constants(self):
        transform = ImageNetInverseTransform()
        expected_mean = torch.tensor(constants.IMAGENET_MEAN)
        expected_std = torch.tensor(constants.IMAGENET_STD)
        self.assertTrue(torch.allclose(transform.mean, expected_mean))
        self.assertTrue(torch.allclose(transform.std, expected_std))

    def test_forward(self):
        transform = ImageNetInverseTransform()
        img = torch.zeros(3, 8, 8)
        result = transform(img)
        # 0 * std + mean = mean; broadcast over HW
        expected = torch.tensor(constants.IMAGENET_MEAN)[:, None, None].expand(3, 8, 8)
        self.assertTrue(torch.allclose(result, expected, atol=1e-5))


class TestDivideBy255(unittest.TestCase):

    def test_divides_correctly(self):
        transform = DivideBy255()
        img = torch.full((3, 4, 4), 255.0)
        result = transform(img)
        self.assertAlmostEqual(result.mean().item(), 1.0, delta=1e-5)

    def test_zero_remains_zero(self):
        transform = DivideBy255()
        img = torch.zeros(3, 4, 4)
        result = transform(img)
        self.assertAlmostEqual(result.mean().item(), 0.0, delta=1e-5)


class TestMulticlassSegmentationTargetTransform(unittest.TestCase):

    def test_output_shape(self):
        num_classes = 3
        transform = MulticlassSegmentationTargetTransform(num_classes)
        target = torch.randint(0, num_classes, (8, 8))  # H,W
        result = transform(target)
        self.assertEqual(result.shape, (num_classes, 8, 8))

    def test_one_hot_encoding(self):
        num_classes = 3
        transform = MulticlassSegmentationTargetTransform(num_classes)
        # Single pixel, all class 2
        target = torch.full((4, 4), 2, dtype=torch.int64)
        result = transform(target)
        # Channel 2 should be all 1s, others 0s
        self.assertTrue(torch.all(result[2] == 1.0))
        self.assertTrue(torch.all(result[0] == 0.0))
        self.assertTrue(torch.all(result[1] == 0.0))

    def test_dtype_is_float(self):
        transform = MulticlassSegmentationTargetTransform(4)
        target = torch.randint(0, 4, (6, 6))
        result = transform(target)
        self.assertEqual(result.dtype, torch.float32)

    def test_asserts_2d_input(self):
        transform = MulticlassSegmentationTargetTransform(3)
        with self.assertRaises(AssertionError):
            transform(torch.randint(0, 3, (2, 8, 8)))  # 3D, should fail


class TestConstants(unittest.TestCase):

    def test_imagenet_mean_length(self):
        self.assertEqual(len(constants.IMAGENET_MEAN), 3)

    def test_imagenet_std_length(self):
        self.assertEqual(len(constants.IMAGENET_STD), 3)

    def test_imagenet_mean_values(self):
        self.assertAlmostEqual(constants.IMAGENET_MEAN[0], 0.485, delta=1e-5)
        self.assertAlmostEqual(constants.IMAGENET_MEAN[1], 0.456, delta=1e-5)
        self.assertAlmostEqual(constants.IMAGENET_MEAN[2], 0.406, delta=1e-5)

    def test_imagenet_std_values(self):
        self.assertAlmostEqual(constants.IMAGENET_STD[0], 0.229, delta=1e-5)
        self.assertAlmostEqual(constants.IMAGENET_STD[1], 0.224, delta=1e-5)
        self.assertAlmostEqual(constants.IMAGENET_STD[2], 0.225, delta=1e-5)


if __name__ == "__main__":
    unittest.main()
