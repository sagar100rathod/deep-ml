import time

import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data.distributed import DistributedSampler

from deepml.accelerator_trainer import AcceleratorTrainer
from deepml.metrics.classification import Accuracy


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

start_time = time.time()
trainer = AcceleratorTrainer(model, optimizer, criterion, {})
trainer.fit(train_loader, val_loader, epochs=10, metrics={"acc": Accuracy()})

end_time = time.time()

print("Time taken (sec):", (end_time - start_time))
