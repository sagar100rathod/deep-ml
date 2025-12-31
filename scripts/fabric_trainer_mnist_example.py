import sys
import time

import torch
import torch.nn.functional as F
import torchvision

sys.path.append("..")

from deepml.fabric_trainer import FabricTrainer
from deepml.metrics.classification import Accuracy
from deepml.tasks import ImageClassification
from deepml.tracking import MLFlowLogger


class MnistModel(torch.nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 10):
        super().__init__()

        self.conv1 = torch.nn.Conv2d(in_channels, 8, kernel_size=(5, 5))
        self.conv2 = torch.nn.Conv2d(8, 16, kernel_size=(5, 5))
        self.linear = torch.nn.Linear(in_features=6400, out_features=num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        x = F.gelu(x)
        x = x.flatten(start_dim=1)
        return self.linear(x)


if __name__ == "__main__":

    devices = 4  # number of devices to use, can be 1, 2, 4, etc.
    accelerator = "cpu"  # "cpu" or "cuda" for GPU training

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
    lr_scheduler_fn = lambda optimizer: torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=10
    )
    criterion = torch.nn.CrossEntropyLoss()
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=16, num_workers=0, shuffle=True, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=16, num_workers=0, shuffle=False, drop_last=True
    )

    print("Train Samples:", len(train_dataset))
    print("Val Samples:", len(test_dataset))
    print("Batch Size:", train_loader.batch_size)
    print("Train Iterations in an Epoch:", len(train_loader))
    print("Val Iterations in an Epoch:", len(val_loader))

    start_time = time.time()
    classification = ImageClassification(
        model, model_dir="./temp/fabric_trainer/model_weights", classes=list(range(10))
    )
    learner = FabricTrainer(
        classification,
        optimizer,
        criterion,
        devices=devices,
        accelerator=accelerator,
        precision="32-true",
        lr_scheduler_fn=lr_scheduler_fn,
        lr_scheduler_step_policy="epoch",
    )

    # sanity test
    if learner.fabric.is_global_zero:
        X, y = val_loader.dataset[0]
        model.eval()
        with torch.no_grad():
            y_pred = model(X.unsqueeze(0))
        print(y_pred)

    learner.fit(
        train_loader,
        val_loader,
        epochs=10,
        metrics={"acc": Accuracy()},
        gradient_accumulation_steps=4,
        logger=MLFlowLogger(),
    )

    if learner.fabric.is_global_zero:
        end_time = time.time()
        print("Time taken (sec):", (end_time - start_time))
