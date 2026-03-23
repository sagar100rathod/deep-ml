import sys
import time

import torch
import torch.nn.functional as F
import torchvision

sys.path.append("..")

from deepml.accelerator_trainer import AcceleratorTrainer
from deepml.metrics.classification import Accuracy
from deepml.tasks import ImageClassification


class MnistModel(torch.nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 10):
        super().__init__()

        self.conv1 = torch.nn.Conv2d(in_channels, 8, kernel_size=(5, 5))
        self.conv2 = torch.nn.Conv2d(8, 16, kernel_size=(5, 5))
        self.linear = torch.nn.LazyLinear(out_features=num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        x = F.gelu(x)
        x = x.flatten(start_dim=1)
        return self.linear(x)


def main():
    torch.manual_seed(123)
    transform = torchvision.transforms.ToTensor()

    train_dataset = torchvision.datasets.MNIST(
        "./data/", download=True, train=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        "./data/", download=True, train=False, transform=transform
    )

    model = MnistModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=16, num_workers=0, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, num_workers=0)

    print("Train Samples:", len(train_dataset))
    print("Val Samples:", len(test_dataset))
    print("Batch Size:", train_loader.batch_size)

    start_time = time.time()
    classification_task = ImageClassification(
        model, model_dir="./accelerator", classes=list(range(10))
    )
    trainer = AcceleratorTrainer(
        classification_task,
        optimizer,
        criterion,
        accelerator_config={"gradient_accumulation_steps": 2},
    )

    trainer.fit(train_loader, val_loader, epochs=10, metrics={"acc": Accuracy()})

    end_time = time.time()

    print("Time taken (sec):", (end_time - start_time))


if __name__ == "__main__":
    main()
