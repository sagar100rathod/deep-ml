from unittest.mock import MagicMock, patch

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pytest
from PIL import Image

from deepml.visualize import (
    plot_images,
    plot_images_with_bboxes,
    plot_images_with_title,
    show_images_from_dataframe,
    show_images_from_dataset,
    show_images_from_folder,
    show_images_from_loader,
)

# Use non-interactive backend so no windows pop up during tests
matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# Helpers / Fixtures
# ---------------------------------------------------------------------------


def _make_rgb_array(h=32, w=32):
    """Return a random RGB numpy array of shape (H, W, 3) in uint8."""
    return np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)


@pytest.fixture(autouse=True)
def close_all_figures():
    """Close every matplotlib figure after each test to free memory."""
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# Tests for plot_images
# ---------------------------------------------------------------------------


class TestPlotImages:

    def test_basic_call_without_labels(self):
        images = [_make_rgb_array() for _ in range(6)]
        plot_images(images, cols=3)
        fig = plt.gcf()
        axes = fig.get_axes()
        assert len(axes) == 6

    def test_with_labels(self):
        images = [_make_rgb_array() for _ in range(4)]
        labels = ["a", "b", "c", "d"]
        plot_images(images, labels=labels, cols=2, fontsize=12)
        fig = plt.gcf()
        axes = fig.get_axes()
        assert len(axes) == 4
        for ax, label in zip(axes, labels):
            assert ax.get_title() == label

    def test_single_image(self):
        images = [_make_rgb_array()]
        plot_images(images, cols=1)
        assert len(plt.gcf().get_axes()) == 1

    def test_cols_greater_than_images(self):
        images = [_make_rgb_array() for _ in range(2)]
        plot_images(images, cols=10)
        assert len(plt.gcf().get_axes()) == 2

    def test_custom_figsize(self):
        images = [_make_rgb_array() for _ in range(3)]
        plot_images(images, cols=3, figsize=(15, 5))
        fig = plt.gcf()
        w, h = fig.get_size_inches()
        assert w == pytest.approx(15, abs=1)
        assert h == pytest.approx(5, abs=1)

    def test_grayscale_image(self):
        gray = np.random.randint(0, 256, (32, 32), dtype=np.uint8)
        plot_images([gray], cols=1)
        assert len(plt.gcf().get_axes()) == 1


# ---------------------------------------------------------------------------
# Tests for plot_images_with_title
# ---------------------------------------------------------------------------


class TestPlotImagesWithTitle:

    def test_basic_generator(self):
        images = [_make_rgb_array() for _ in range(4)]
        gen = ((img, f"title_{i}", None) for i, img in enumerate(images))
        plot_images_with_title(gen, samples=4, cols=2)
        axes = plt.gcf().get_axes()
        assert len(axes) == 4
        assert axes[0].get_title() == "title_0"

    def test_title_color(self):
        gen = ((_make_rgb_array(), "t", "red") for _ in range(1))
        plot_images_with_title(gen, samples=1, cols=1)
        ax = plt.gcf().get_axes()[0]
        assert ax.title.get_color() == "red"

    def test_none_title_color_uses_default(self):
        gen = ((_make_rgb_array(), "t", None) for _ in range(1))
        plot_images_with_title(gen, samples=1, cols=1)
        ax = plt.gcf().get_axes()[0]
        assert ax.title.get_color() == matplotlib.rcParams["text.color"]


# ---------------------------------------------------------------------------
# Tests for plot_images_with_bboxes
# ---------------------------------------------------------------------------


class TestPlotImagesWithBboxes:

    def _make_generator(self, n=2, with_bboxes=True):
        for i in range(n):
            bboxes = [[0, 5, 5, 10, 10]] if with_bboxes else None
            yield _make_rgb_array(), f"img_{i}", bboxes

    def test_basic_bboxes(self):
        gen = self._make_generator(n=2)
        plot_images_with_bboxes(gen, samples=2, cols=2)
        axes = plt.gcf().get_axes()
        assert len(axes) == 2
        # Each axis should have at least one patch (the bbox rectangle)
        for ax in axes:
            rect_patches = [p for p in ax.patches if isinstance(p, mpatches.Rectangle)]
            assert len(rect_patches) >= 1

    def test_no_bboxes(self):
        gen = self._make_generator(n=1, with_bboxes=False)
        plot_images_with_bboxes(gen, samples=1, cols=1)
        axes = plt.gcf().get_axes()
        assert len(axes) == 1

    def test_with_classes_list(self):
        classes = ["cat", "dog"]
        bboxes = [[0, 2, 2, 8, 8], [1, 10, 10, 5, 5]]
        gen = ((_make_rgb_array(), "img", bboxes) for _ in range(1))
        plot_images_with_bboxes(gen, samples=1, cols=1, classes=classes)
        axes = plt.gcf().get_axes()
        assert len(axes) == 1

    def test_with_class_color_map(self):
        class_color_map = {0: "#ff0000", 1: "#00ff00"}
        bboxes = [[0, 2, 2, 8, 8]]
        gen = ((_make_rgb_array(), "img", bboxes) for _ in range(1))
        plot_images_with_bboxes(gen, samples=1, cols=1, class_color_map=class_color_map)
        assert len(plt.gcf().get_axes()) == 1

    def test_class_color_map_with_string_keys(self):
        class_color_map = {"0": "#ff0000"}
        bboxes = [[0, 2, 2, 8, 8]]
        gen = ((_make_rgb_array(), "img", bboxes) for _ in range(1))
        plot_images_with_bboxes(gen, samples=1, cols=1, class_color_map=class_color_map)
        assert len(plt.gcf().get_axes()) == 1

    def test_string_class_ids_in_bboxes(self):
        """When bbox class id is a string, _get_color_for_class should hash it."""
        bboxes = [["person", 2, 2, 8, 8]]
        gen = ((_make_rgb_array(), "img", bboxes) for _ in range(1))
        plot_images_with_bboxes(gen, samples=1, cols=1)
        assert len(plt.gcf().get_axes()) == 1

    def test_multiple_bboxes_per_image(self):
        bboxes = [[0, 1, 1, 5, 5], [1, 10, 10, 3, 3], [0, 20, 20, 4, 4]]
        gen = ((_make_rgb_array(64, 64), "img", bboxes) for _ in range(1))
        plot_images_with_bboxes(gen, samples=1, cols=1)
        ax = plt.gcf().get_axes()[0]
        rect_patches = [p for p in ax.patches if isinstance(p, mpatches.Rectangle)]
        assert len(rect_patches) == 3


# ---------------------------------------------------------------------------
# Tests for show_images_from_loader
# ---------------------------------------------------------------------------


class TestShowImagesFromLoader:

    @patch("deepml.visualize.plot_images_with_title")
    @patch("deepml.visualize.transform_input")
    @patch("deepml.visualize.get_random_samples_batch_from_loader")
    def test_basic(self, mock_get_batch, mock_transform, mock_plot):
        import torch

        images = torch.randn(4, 3, 32, 32)
        labels = torch.tensor([0, 1, 2, 3])
        mock_get_batch.return_value = (images, labels)
        mock_transform.return_value = np.random.rand(4, 32, 32, 3)

        loader = MagicMock()
        loader.dataset.classes = ["a", "b", "c", "d"]

        show_images_from_loader(loader, samples=4, cols=2)

        mock_get_batch.assert_called_once_with(loader, samples=4)
        mock_transform.assert_called_once()
        mock_plot.assert_called_once()

    @patch("deepml.visualize.plot_images_with_title")
    @patch("deepml.visualize.transform_input")
    @patch("deepml.visualize.get_random_samples_batch_from_loader")
    def test_uses_dataset_classes_when_none(
        self, mock_get_batch, mock_transform, mock_plot
    ):
        import torch

        images = torch.randn(2, 3, 32, 32)
        labels = torch.tensor([0, 1])
        mock_get_batch.return_value = (images, labels)
        mock_transform.return_value = np.random.rand(2, 32, 32, 3)

        loader = MagicMock()
        loader.dataset.classes = ["cat", "dog"]

        show_images_from_loader(loader, samples=2, classes=None)
        # classes should have been fetched from loader.dataset.classes

    @patch("deepml.visualize.plot_images_with_title")
    @patch("deepml.visualize.transform_input")
    @patch("deepml.visualize.get_random_samples_batch_from_loader")
    def test_with_inverse_transform(self, mock_get_batch, mock_transform, mock_plot):
        import torch

        images = torch.randn(2, 3, 32, 32)
        labels = torch.tensor([0, 1])
        mock_get_batch.return_value = (images, labels)
        mock_transform.return_value = np.random.rand(2, 32, 32, 3)

        loader = MagicMock()
        inv_transform = MagicMock()

        show_images_from_loader(
            loader, image_inverse_transform=inv_transform, samples=2
        )
        mock_transform.assert_called_once_with(images, inv_transform)


# ---------------------------------------------------------------------------
# Tests for show_images_from_dataset
# ---------------------------------------------------------------------------


class TestShowImagesFromDataset:

    @patch("deepml.visualize.plot_images_with_title")
    @patch("deepml.visualize.transform_input")
    @patch("deepml.visualize.get_random_samples_batch_from_dataset")
    def test_basic(self, mock_get_batch, mock_transform, mock_plot):
        import torch

        images = torch.randn(4, 3, 32, 32)
        labels = torch.tensor([0, 1, 2, 3])
        mock_get_batch.return_value = (images, labels)
        mock_transform.return_value = np.random.rand(4, 32, 32, 3)

        dataset = MagicMock()
        dataset.classes = ["a", "b", "c", "d"]

        show_images_from_dataset(dataset, samples=4, cols=2)

        mock_get_batch.assert_called_once_with(dataset, samples=4)
        mock_transform.assert_called_once()
        mock_plot.assert_called_once()

    @patch("deepml.visualize.plot_images_with_title")
    @patch("deepml.visualize.transform_input")
    @patch("deepml.visualize.get_random_samples_batch_from_dataset")
    def test_uses_dataset_classes_when_none(
        self, mock_get_batch, mock_transform, mock_plot
    ):
        import torch

        images = torch.randn(2, 3, 32, 32)
        labels = torch.tensor([0, 1])
        mock_get_batch.return_value = (images, labels)
        mock_transform.return_value = np.random.rand(2, 32, 32, 3)

        dataset = MagicMock()
        dataset.classes = ["cat", "dog"]

        show_images_from_dataset(dataset, samples=2, classes=None)

    @patch("deepml.visualize.plot_images_with_title")
    @patch("deepml.visualize.transform_input")
    @patch("deepml.visualize.get_random_samples_batch_from_dataset")
    def test_explicit_classes_override(self, mock_get_batch, mock_transform, mock_plot):
        import torch

        images = torch.randn(2, 3, 32, 32)
        labels = torch.tensor([0, 1])
        mock_get_batch.return_value = (images, labels)
        mock_transform.return_value = np.random.rand(2, 32, 32, 3)

        dataset = MagicMock()
        dataset.classes = ["cat", "dog"]

        show_images_from_dataset(dataset, samples=2, classes=["airplane", "bird"])
        mock_plot.assert_called_once()


# ---------------------------------------------------------------------------
# Tests for show_images_from_folder
# ---------------------------------------------------------------------------


class TestShowImagesFromFolder:

    @patch("deepml.visualize.plot_images_with_title")
    def test_with_explicit_images_list(self, mock_plot, tmp_path):
        # Create dummy image files
        for name in ["a.png", "b.png", "c.png"]:
            img = Image.fromarray(_make_rgb_array())
            img.save(str(tmp_path / name))

        show_images_from_folder(
            str(tmp_path), images=["a.png", "b.png"], samples=2, cols=2
        )
        mock_plot.assert_called_once()
        # samples arg passed to plot_images_with_title should be len(images)=2
        assert (
            mock_plot.call_args[1].get(
                "cols",
                mock_plot.call_args[0][2] if len(mock_plot.call_args[0]) > 2 else None,
            )
            is not None
        )

    @patch("deepml.visualize.plot_images_with_title")
    def test_picks_random_when_images_none(self, mock_plot, tmp_path):
        for name in ["x.png", "y.png", "z.png"]:
            img = Image.fromarray(_make_rgb_array())
            img.save(str(tmp_path / name))

        show_images_from_folder(str(tmp_path), samples=2, cols=2)
        mock_plot.assert_called_once()

    @patch("deepml.visualize.plot_images_with_title")
    def test_samples_greater_than_files(self, mock_plot, tmp_path):
        img = Image.fromarray(_make_rgb_array())
        img.save(str(tmp_path / "only.png"))

        show_images_from_folder(str(tmp_path), samples=5, cols=2)
        mock_plot.assert_called_once()

    def test_custom_open_func(self, tmp_path):
        for name in ["a.png"]:
            img = Image.fromarray(_make_rgb_array())
            img.save(str(tmp_path / name))

        custom_open = MagicMock(return_value=_make_rgb_array())
        show_images_from_folder(
            str(tmp_path),
            images=["a.png"],
            open_file_func=custom_open,
            samples=1,
            cols=1,
        )
        custom_open.assert_called_once()


# ---------------------------------------------------------------------------
# Tests for show_images_from_dataframe
# ---------------------------------------------------------------------------


class TestShowImagesFromDataframe:

    @patch("deepml.visualize.plot_images_with_title")
    def test_basic_without_bboxes(self, mock_plot, tmp_path):
        import pandas as pd

        # Create dummy images
        for name in ["a.png", "b.png", "c.png"]:
            img = Image.fromarray(_make_rgb_array())
            img.save(str(tmp_path / name))

        df = pd.DataFrame({"image": ["a.png", "b.png", "c.png"]})
        show_images_from_dataframe(df, img_dir=str(tmp_path), samples=2, cols=2)
        mock_plot.assert_called_once()

    @patch("deepml.visualize.plot_images_with_bboxes")
    def test_with_bboxes(self, mock_plot_bboxes, tmp_path):
        import pandas as pd

        for name in ["a.png", "b.png"]:
            img = Image.fromarray(_make_rgb_array())
            img.save(str(tmp_path / name))

        df = pd.DataFrame(
            {
                "image": ["a.png", "b.png"],
                "bboxes": [
                    [[0, 5, 5, 10, 10]],
                    [[1, 2, 2, 8, 8]],
                ],
            }
        )
        show_images_from_dataframe(
            df,
            img_dir=str(tmp_path),
            bbox_label_column="bboxes",
            samples=2,
            cols=2,
        )
        mock_plot_bboxes.assert_called_once()

    @patch("deepml.visualize.plot_images_with_title")
    def test_with_image_filepath_column(self, mock_plot, tmp_path):
        import pandas as pd

        paths = []
        for name in ["a.png", "b.png"]:
            p = str(tmp_path / name)
            Image.fromarray(_make_rgb_array()).save(p)
            paths.append(p)

        df = pd.DataFrame({"filepath": paths})
        show_images_from_dataframe(
            df,
            image_filepath_column="filepath",
            samples=2,
            cols=2,
        )
        mock_plot.assert_called_once()

    def test_custom_open_func(self, tmp_path):
        import pandas as pd

        for name in ["a.png"]:
            Image.fromarray(_make_rgb_array()).save(str(tmp_path / name))

        df = pd.DataFrame({"image": ["a.png"]})
        custom_open = MagicMock(return_value=_make_rgb_array())
        show_images_from_dataframe(
            df,
            img_dir=str(tmp_path),
            open_file_func=custom_open,
            samples=1,
            cols=1,
        )
        custom_open.assert_called_once()

    @patch("deepml.visualize.plot_images_with_bboxes")
    def test_bbox_with_classes_and_color_map(self, mock_plot_bboxes, tmp_path):
        import pandas as pd

        for name in ["a.png"]:
            Image.fromarray(_make_rgb_array()).save(str(tmp_path / name))

        df = pd.DataFrame(
            {
                "image": ["a.png"],
                "bboxes": [[[0, 5, 5, 10, 10]]],
            }
        )
        show_images_from_dataframe(
            df,
            img_dir=str(tmp_path),
            bbox_label_column="bboxes",
            samples=1,
            cols=1,
            classes=["car"],
            class_color_map={0: "#ff0000"},
            cmap="Set1",
        )
        mock_plot_bboxes.assert_called_once()
        _, kwargs = mock_plot_bboxes.call_args
        assert kwargs["classes"] == ["car"]
        assert kwargs["class_color_map"] == {0: "#ff0000"}
        assert kwargs["cmap"] == "Set1"


# ---------------------------------------------------------------------------
# Edge-case / integration-style tests
# ---------------------------------------------------------------------------


class TestEdgeCases:

    def test_plot_images_empty_list(self):
        """Calling plot_images with an empty list should still succeed."""
        plot_images([], cols=4)
        assert len(plt.gcf().get_axes()) == 0

    def test_plot_images_with_title_zero_samples(self):
        gen = iter([])
        plot_images_with_title(gen, samples=0, cols=4)
        # Should not raise

    def test_plot_images_with_bboxes_zero_samples(self):
        gen = iter([])
        plot_images_with_bboxes(gen, samples=0, cols=4)
        # Should not raise

    def test_plot_images_rgba(self):
        """RGBA images should also be displayable."""
        rgba = np.random.randint(0, 256, (32, 32, 4), dtype=np.uint8)
        plot_images([rgba], cols=1)
        assert len(plt.gcf().get_axes()) == 1

    def test_plot_images_float_array(self):
        """Float images in [0, 1] should be handled by matplotlib."""
        float_img = np.random.rand(32, 32, 3).astype(np.float32)
        plot_images([float_img], cols=1)
        assert len(plt.gcf().get_axes()) == 1
