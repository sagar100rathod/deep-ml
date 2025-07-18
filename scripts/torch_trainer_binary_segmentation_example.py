import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image, ImageDraw

sys.path.append("..")
from deepml.metrics.segmentation import Precision, Recall
from deepml.model_arch.unet import ResNetUNet
from deepml.tasks import Segmentation
from deepml.tracking import MLFlowLogger
from deepml.trainer import Learner


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

        color = tuple(np.random.randint(64, 256, size=3))
        label = 1

        # only triangles for simplicity
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
        mask = torch.from_numpy(np.array(mask)).unsqueeze(0).to(torch.int64)

        return img, mask


class BinaryBCELogitsLoss(torch.nn.Module):
    def __init__(self):
        super(BinaryBCELogitsLoss, self).__init__()
        self.loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets):
        # Ensure inputs are logits
        return self.loss(
            inputs, targets.to(torch.float32)
        )  # Remove channel dimension for binary segmentation


if __name__ == "__main__":

    device = "cpu"

    torch.manual_seed(123)
    transform = torchvision.transforms.ToTensor()

    train_dataset = SyntheticShapesDataset(
        length=300, img_size=128, transform=transform
    )
    val_dataset = SyntheticShapesDataset(length=100, img_size=128, transform=transform)

    model = ResNetUNet(n_class=1)  # ['background', 'triangle']
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    criterion = BinaryBCELogitsLoss()
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=5, num_workers=0, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=5, num_workers=0)

    X, y = val_loader.dataset[0]
    print("Val Input (X, y):", X.shape, y.shape)

    start_time = time.time()
    segmentation = Segmentation(
        model,
        model_dir="./temp/torch_trainer/seg_model_weights",
        num_classes=1,
        device=device,
    )

    learner = Learner(segmentation, optimizer, criterion)

    logger = MLFlowLogger()
    learner.fit(
        train_loader,
        val_loader,
        epochs=10,
        metrics={
            "precision": Precision(mode="binary", reduction="macro", threshold=0.5),
            "recall": Recall(mode="binary", reduction="macro", threshold=0.5),
        },
        gradient_accumulation_steps=1,
        logger=logger,
        logger_img_size=224,
    )

    end_time = time.time()
    print("Time taken (sec):", (end_time - start_time))
