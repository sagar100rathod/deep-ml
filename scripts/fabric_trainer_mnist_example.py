import time

import torch
import torch.nn.functional as F
import torchvision

from deepml.fabric_trainer import FabricTrainer
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


def train_mnist(devices: int = 1, accelerator: str = "cpu"):

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

    X, y = train_dataset[0]

    # need to initialize lazy layers in torch before using multiple devices in fabric
    model.eval()
    with torch.no_grad():
        y_pred = model(X.unsqueeze(0))
        # print(y_pred)

    start_time = time.time()
    classification = ImageClassification(
        model, model_dir="../deepml/temp/deepml/model_weights", classes=list(range(10))
    )
    learner = FabricTrainer(
        classification, optimizer, criterion, devices=devices, accelerator=accelerator
    )

    learner.fit(
        train_loader,
        val_loader,
        epochs=10,
        metrics={"acc": Accuracy()},
        gradient_accumulation_steps=4,
    )

    end_time = time.time()
    print("Time taken (sec):", (end_time - start_time))


if __name__ == "__main__":
    train_mnist(devices=4, accelerator="cpu")
