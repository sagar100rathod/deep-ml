"""
Tests for Segmentation.show_predictions with overlay support.

Covers:
  - binary mode: shows input, target mask, target overlay, pred mask, pred overlay
  - multiclass mode: same 5-column layout with RGB masks
  - overlay tensors are uint8 BHWC with correct spatial dimensions
  - plot_images is called with 5 titles per sample
"""

from unittest.mock import patch

import torch
import torch.nn as nn

from deepml.tasks import Segmentation

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

H, W = 32, 32
BATCH = 2


class _BinarySegModel(nn.Module):
    """Always predicts 'foreground' (large positive logit)."""

    def forward(self, x):
        B = x.shape[0]
        return torch.ones(B, 1, H, W) * 5.0  # sigmoid → ~1.0


class _MulticlassSegModel(nn.Module):
    """3-class model; always predicts class 1."""

    def forward(self, x):
        B = x.shape[0]
        logits = torch.zeros(B, 3, H, W)
        logits[:, 1, :, :] = 10.0  # argmax → class 1
        return logits


def _make_loader(num_classes: int = 1):
    """Returns a DataLoader with a single batch of (images, masks)."""
    images = torch.rand(BATCH, 3, H, W)
    if num_classes == 1:
        masks = torch.randint(0, 2, (BATCH, H, W)).long()
    else:
        masks = torch.randint(0, num_classes, (BATCH, H, W)).long()
    dataset = torch.utils.data.TensorDataset(images, masks)
    return torch.utils.data.DataLoader(dataset, batch_size=BATCH)


def _make_task(model, mode="binary", num_classes=1, color_map=None):
    return Segmentation(
        model=model,
        model_dir="/tmp/seg_test",
        mode=mode,
        num_classes=num_classes,
        device="cpu",
        color_map=color_map,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestShowPredictionsBinary:

    def setup_method(self):
        self.task = _make_task(_BinarySegModel(), mode="binary", num_classes=1)
        self.loader = _make_loader(num_classes=1)

    @patch("deepml.tasks.plot_images")
    @patch("deepml.tasks.get_random_samples_batch_from_loader")
    def test_plot_images_called_with_5_cols(self, mock_loader, mock_plot):
        images_batch = torch.rand(BATCH, 3, H, W)
        masks_batch = torch.randint(0, 2, (BATCH, H, W)).long()
        mock_loader.return_value = (images_batch, masks_batch)

        self.task.show_predictions(self.loader)

        mock_plot.assert_called_once()
        _, kwargs = mock_plot.call_args[0], mock_plot.call_args[1]
        positional = mock_plot.call_args[0]
        # positional[0] = images list, positional[1] = titles list
        images_list = positional[0]
        titles_list = positional[1]

        # 5 images per sample
        assert len(images_list) == BATCH * 5
        assert len(titles_list) == BATCH * 5

    @patch("deepml.tasks.plot_images")
    @patch("deepml.tasks.get_random_samples_batch_from_loader")
    def test_titles_contain_expected_labels(self, mock_loader, mock_plot):
        images_batch = torch.rand(BATCH, 3, H, W)
        masks_batch = torch.randint(0, 2, (BATCH, H, W)).long()
        mock_loader.return_value = (images_batch, masks_batch)

        self.task.show_predictions(self.loader)

        titles = mock_plot.call_args[0][1]
        expected = [
            "Input",
            "Target Mask",
            "Target Overlay",
            "Pred Mask",
            "Pred Overlay",
        ]
        assert titles == expected * BATCH

    @patch("deepml.tasks.plot_images")
    @patch("deepml.tasks.get_random_samples_batch_from_loader")
    def test_cols_fixed_at_5(self, mock_loader, mock_plot):
        images_batch = torch.rand(BATCH, 3, H, W)
        masks_batch = torch.randint(0, 2, (BATCH, H, W)).long()
        mock_loader.return_value = (images_batch, masks_batch)

        self.task.show_predictions(self.loader)

        call_kwargs = mock_plot.call_args[1]
        assert call_kwargs.get("cols") == 5

    @patch("deepml.tasks.plot_images")
    @patch("deepml.tasks.get_random_samples_batch_from_loader")
    def test_overlay_shape_matches_input(self, mock_loader, mock_plot):
        """Overlay images should have the same H x W as input and be HWC."""
        images_batch = torch.rand(BATCH, 3, H, W)
        masks_batch = torch.randint(0, 2, (BATCH, H, W)).long()
        mock_loader.return_value = (images_batch, masks_batch)

        self.task.show_predictions(self.loader)

        images_list = mock_plot.call_args[0][0]
        # images_list layout per sample: input(HWC), target_mask(HWC), target_overlay(HWC), pred_mask(HWC), pred_overlay(HWC)
        for sample_idx in range(BATCH):
            base = sample_idx * 5
            input_img = images_list[base]
            target_overlay = images_list[base + 2]
            pred_overlay = images_list[base + 4]

            assert input_img.shape[:2] == (H, W), "Input spatial dims mismatch"
            assert target_overlay.shape[:2] == (
                H,
                W,
            ), "Target overlay spatial dims mismatch"
            assert pred_overlay.shape[:2] == (
                H,
                W,
            ), "Pred overlay spatial dims mismatch"
            # overlays should be RGB (3-channel)
            assert target_overlay.shape[2] == 3, "Target overlay should be RGB"
            assert pred_overlay.shape[2] == 3, "Pred overlay should be RGB"

    @patch("deepml.tasks.plot_images")
    @patch("deepml.tasks.get_random_samples_batch_from_loader")
    def test_masks_are_3channel_after_expansion(self, mock_loader, mock_plot):
        """Binary masks should be expanded to 3 channels for display."""
        images_batch = torch.rand(BATCH, 3, H, W)
        masks_batch = torch.zeros(BATCH, H, W).long()
        mock_loader.return_value = (images_batch, masks_batch)

        self.task.show_predictions(self.loader)

        images_list = mock_plot.call_args[0][0]
        for sample_idx in range(BATCH):
            base = sample_idx * 5
            target_mask = images_list[base + 1]
            pred_mask = images_list[base + 3]
            assert target_mask.shape[2] == 3
            assert pred_mask.shape[2] == 3


class TestShowPredictionsMulticlass:

    def setup_method(self):
        color_map = {0: [0, 0, 0], 1: [255, 0, 0], 2: [0, 255, 0]}
        self.task = _make_task(
            _MulticlassSegModel(),
            mode="multiclass",
            num_classes=3,
            color_map=color_map,
        )
        self.loader = _make_loader(num_classes=3)

    @patch("deepml.tasks.plot_images")
    @patch("deepml.tasks.get_random_samples_batch_from_loader")
    def test_plot_images_called_with_5_images_per_sample(self, mock_loader, mock_plot):
        images_batch = torch.rand(BATCH, 3, H, W)
        masks_batch = torch.randint(0, 3, (BATCH, H, W)).long()
        mock_loader.return_value = (images_batch, masks_batch)

        self.task.show_predictions(self.loader)

        images_list = mock_plot.call_args[0][0]
        assert len(images_list) == BATCH * 5

    @patch("deepml.tasks.plot_images")
    @patch("deepml.tasks.get_random_samples_batch_from_loader")
    def test_overlay_is_rgb(self, mock_loader, mock_plot):
        images_batch = torch.rand(BATCH, 3, H, W)
        masks_batch = torch.randint(0, 3, (BATCH, H, W)).long()
        mock_loader.return_value = (images_batch, masks_batch)

        self.task.show_predictions(self.loader)

        images_list = mock_plot.call_args[0][0]
        for sample_idx in range(BATCH):
            base = sample_idx * 5
            assert images_list[base + 2].shape[2] == 3  # target overlay RGB
            assert images_list[base + 4].shape[2] == 3  # pred overlay RGB

    @patch("deepml.tasks.plot_images")
    @patch("deepml.tasks.get_random_samples_batch_from_loader")
    def test_titles_multiclass(self, mock_loader, mock_plot):
        images_batch = torch.rand(BATCH, 3, H, W)
        masks_batch = torch.randint(0, 3, (BATCH, H, W)).long()
        mock_loader.return_value = (images_batch, masks_batch)

        self.task.show_predictions(self.loader)

        titles = mock_plot.call_args[0][1]
        expected = [
            "Input",
            "Target Mask",
            "Target Overlay",
            "Pred Mask",
            "Pred Overlay",
        ]
        assert titles == expected * BATCH
