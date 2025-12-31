import random
import sys
import time

import numpy as np
import torch
import torchvision
from PIL import Image, ImageDraw

sys.path.append("..")
from deepml.fabric_trainer import FabricTrainer
from deepml.metrics.segmentation import Precision, Recall
from deepml.model_arch.unet import ResNetUNet
from deepml.tasks import Segmentation
from deepml.tracking import MLFlowLogger


class SyntheticShapesDataset(torch.utils.data.Dataset):
    def __init__(self, length=1000, img_size=128, transform=None):
        """
        Args:
            length: number of samples in the dataset (virtual)
            img_size: image size (square)
            transform: torchvision transforms for the image
        """
        self.length = length
        self.img_size = img_size
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Create black background
        img = Image.new("RGB", (self.img_size, self.img_size), (0, 0, 0))
        mask = Image.new("L", (self.img_size, self.img_size), 0)

        draw = ImageDraw.Draw(img)
        mask_draw = ImageDraw.Draw(mask)

        shape_type = random.choice(["circle", "square", "triangle"])
        color = tuple(np.random.randint(64, 256, size=3))
        label = {"circle": 1, "square": 2, "triangle": 3}[shape_type]

        # Random position & size
        x0 = random.randint(10, self.img_size // 2)
        y0 = random.randint(10, self.img_size // 2)
        x1 = random.randint(self.img_size // 2, self.img_size - 10)
        y1 = random.randint(self.img_size // 2, self.img_size - 10)

        if shape_type == "circle":
            draw.ellipse([x0, y0, x1, y1], fill=color)
            mask_draw.ellipse([x0, y0, x1, y1], fill=label)
        elif shape_type == "square":
            draw.rectangle([x0, y0, x1, y1], fill=color)
            mask_draw.rectangle([x0, y0, x1, y1], fill=label)
        elif shape_type == "triangle":
            points = [
                (
                    random.randint(10, self.img_size - 10),
                    random.randint(10, self.img_size - 10),
                )
                for _ in range(3)
            ]
            draw.polygon(points, fill=color)
            mask_draw.polygon(points, fill=label)

        # Apply transforms
        if self.transform:
            img = self.transform(img)

        # Convert mask to tensor: shape (H, W)
        mask = torch.from_numpy(np.array(mask, dtype=np.int64))

        return img, mask


if __name__ == "__main__":

    device = "cpu"

    torch.manual_seed(123)
    transform = torchvision.transforms.ToTensor()

    train_dataset = SyntheticShapesDataset(
        length=500, img_size=128, transform=transform
    )
    val_dataset = SyntheticShapesDataset(length=50, img_size=128, transform=transform)

    model = ResNetUNet(n_class=4)  # ['background', 'circle', 'square', 'triangle']
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=8, num_workers=0, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=8, num_workers=0)

    X, y = val_loader.dataset[0]
    print("Val Input (X, y):", X.shape, y.shape)

    start_time = time.time()
    segmentation = Segmentation(
        model,
        model_dir="./temp/torch_trainer/seg_model_weights",
        mode="multiclass",
        num_classes=30,
        device=device,
    )

    learner = FabricTrainer(segmentation, optimizer, criterion, accelerator="auto")

    logger = MLFlowLogger()
    learner.fit(
        train_loader,
        val_loader,
        epochs=50,
        metrics={
            "precision": Precision(mode="multiclass", num_classes=4),
            "recall": Recall(mode="multiclass", num_classes=4),
            "precision_circle": Precision(
                mode="multiclass",
                num_classes=4,
                target_class_index=1,
            ),
            "precision_square": Precision(
                mode="multiclass",
                num_classes=4,
                target_class_index=2,
            ),
            "precision_triangle": Precision(
                mode="multiclass",
                num_classes=4,
                target_class_index=3,
            ),
            "recall_circle": Recall(
                mode="multiclass",
                num_classes=4,
                target_class_index=1,
            ),
            "recall_square": Recall(
                mode="multiclass",
                num_classes=4,
                target_class_index=2,
            ),
            "recall_triangle": Recall(
                mode="multiclass",
                num_classes=4,
                target_class_index=3,
            ),
        },
        gradient_accumulation_steps=2,
        logger=logger,
        logger_img_size=128,
    )

    end_time = time.time()
    print("Time taken (sec):", (end_time - start_time))
