# deep-ml

![Licence](https://img.shields.io/badge/License-MIT-green)
![Python](https://img.shields.io/badge/Python-3.8%2B-orange)
[![Downloads](https://static.pepy.tech/personalized-badge/deepml?period=total&units=international_system&left_color=black&right_color=brightgreen&left_text=Downloads)](https://pepy.tech/project/deepml)
![Contributions welcome](https://img.shields.io/badge/Contributions-welcome-yellow)

**deep-ml** is a high-level PyTorch training framework that simplifies deep learning workflows for computer vision tasks. It provides easy-to-use trainers with distributed training support, comprehensive task implementations, and seamless experiment tracking.

## Key Features

### Multiple Training Backends
- **FabricTrainer**: Lightning Fabric for distributed training (recommended for multi-GPU)
- **AcceleratorTrainer**: HuggingFace Accelerate integration (recommended for multi-GPU)
- **Learner**: Classic PyTorch trainer (single-device, notebook-friendly)

### Pre-built Task Implementations
- **Image Classification** (single & multi-label)
- **Semantic Segmentation** (binary & multiclass)
- **Image Regression**
- **Custom tasks** via extensible base classes

### Experiment Tracking
- **TensorBoard** integration (default)
- **MLflow** support
- **Weights & Biases** (wandb) integration
- Custom logger interface

### Advanced Training Features
- ✅ Automatic Mixed Precision (AMP)
- ✅ Gradient accumulation & clipping
- ✅ Learning rate scheduling with warmup
- ✅ Multi-GPU and distributed training
- ✅ Checkpoint management
- ✅ Progress bars and real-time metrics

## Installation

### Basic Installation

```bash
pip install deepml
```

### With Optional Dependencies

```bash
# For Lightning Fabric
pip install deepml lightning-fabric

# For HuggingFace Accelerate
pip install deepml accelerate

# For MLflow tracking
pip install deepml mlflow

# For Weights & Biases
pip install deepml wandb

# For Albumentations (segmentation)
pip install deepml albumentations
```

## Quick Start

### Image Classification

```python
from deepml.tasks import ImageClassification
from deepml.fabric_trainer import FabricTrainer
import torch
from torch.optim import Adam
from torchvision.models import resnet18

# 1. Define your model
model = resnet18(num_classes=10)

# 2. Create a task
task = ImageClassification(
    model=model,
    model_dir="./checkpoints",
    classes=['cat', 'dog', 'bird', ...]  # Optional
)

# 3. Setup optimizer and loss
optimizer = Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()

# 4. Create trainer
trainer = FabricTrainer(
    task=task,
    optimizer=optimizer,
    criterion=criterion,
    accelerator="auto",  # Use GPU if available
    devices="auto",      # Use all available devices
    precision="16-mixed" # Mixed precision training
)

# 5. Train!
trainer.fit(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=50
)

# 6. Visualize predictions
task.show_predictions(loader=val_loader, samples=9)
```

### Semantic Segmentation

```python
from deepml.tasks import Segmentation
from deepml.fabric_trainer import FabricTrainer
from deepml.losses import JaccardLoss

# Define model (e.g., U-Net)
model = UNet(in_channels=3, out_channels=1)

# Create task
task = Segmentation(
    model=model,
    model_dir="./checkpoints",
    mode="binary",
    num_classes=1,
    threshold=0.5
)

# Setup training
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.BCEWithLogitsLoss()

trainer = FabricTrainer(task=task, optimizer=optimizer, criterion=criterion)

# Train
trainer.fit(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=100
)
```

## Documentation

Full documentation is available at: [Documentation Link]

- **Getting Started**: Installation and quick start guide
- **User Guide**: Detailed guides for trainers, tasks, datasets, etc.
- **API Reference**: Complete API documentation
- **Tutorials**: Step-by-step tutorials for common use cases
- **Examples**: Complete example projects

## Tutorials

### Available Tutorials

1. **Image Classification**: Train ResNet on CIFAR-10
2. **Transfer Learning**: Fine-tune pre-trained models
3. **Semantic Segmentation**: U-Net for binary segmentation
4. **Multi-GPU Training**: Distributed training across GPUs
5. **Hyperparameter Tuning**: Optimize with Optuna
6. **Model Deployment**: Export to TorchScript/ONNX

See the [tutorials documentation](docs/source/tutorials.rst) for complete guides.

## 💡 Advanced Features

### Distributed Training

```python
# Multi-GPU training with DDP
trainer = FabricTrainer(
    task=task,
    optimizer=optimizer,
    criterion=criterion,
    accelerator="gpu",
    strategy="ddp",
    devices="auto"  # Use all GPUs
)
```

### Gradient Accumulation

```python
# Simulate larger batch sizes
trainer.fit(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=50,
    gradient_accumulation_steps=4  # Effective batch = 4x
)
```

### Learning Rate Scheduling

```python
from deepml.lr_scheduler_utils import setup_one_cycle_lr_scheduler_with_warmup

lr_scheduler_fn = lambda opt: setup_one_cycle_lr_scheduler_with_warmup(
    optimizer=opt,
    steps_per_epoch=len(train_loader),
    warmup_ratio=0.1,
    num_epochs=50,
    max_lr=1e-3
)

trainer = FabricTrainer(
    ...,
    lr_scheduler_fn=lr_scheduler_fn
)
```

### Experiment Tracking

```python
from deepml.tracking import MLFlowLogger, WandbLogger

# MLflow
logger = MLFlowLogger(
    experiment_name='my-experiment',
    tracking_uri='./mlruns'
)

# Weights & Biases
logger = WandbLogger(
    project='my-project',
    name='experiment-1'
)

trainer.fit(..., logger=logger)
```

## Supported Tasks

| Task | Description | Typical Use Cases |
|------|-------------|-------------------|
| `ImageClassification` | Single-label classification | CIFAR-10, ImageNet |
| `MultiLabelImageClassification` | Multi-label classification | Object attributes |
| `Segmentation` | Pixel-level classification | Medical imaging, autonomous driving |
| `ImageRegression` | Continuous value prediction | Age estimation, depth prediction |
| `NeuralNetTask` | Generic task template | Custom tasks |

## Custom Loss Functions

- **JaccardLoss**: IoU loss for segmentation
- **RMSELoss**: Root mean squared error
- **WeightedBCEWithLogitsLoss**: Weighted binary cross-entropy
- **ContrastiveLoss**: For siamese networks
- **AngularPenaltySMLoss**: ArcFace, SphereFace, CosFace for face recognition

## Metrics

- **Classification**: Accuracy, BinaryAccuracy
- **Segmentation**: IoU, Dice Coefficient, Pixel Accuracy
- **Custom**: Easy to implement custom metrics

## Datasets

- **ImageDataFrameDataset**: Load from pandas DataFrame
- **ImageRowDataFrameDataset**: Flattened arrays in DataFrame
- **SegmentationDataFrameDataset**: Images + masks with Albumentations
- **ImageListDataset**: Directory of images

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](docs/source/contributing.rst) for guidelines.

### Development Setup

```bash
git clone https://github.com/sagar100rathod/deep-ml.git
cd deep-ml
pip install -e ".[dev]"
pytest  # Run tests
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- PyTorch team for the amazing framework
- Lightning AI for Lightning Fabric
- HuggingFace for Accelerate
- All contributors to this project

## Contact

- **Author**: Sagar Rathod
- **Issues**: [GitHub Issues](https://github.com/sagar100rathod/deep-ml/issues)
- **Discussions**: [GitHub Discussions](https://github.com/sagar100rathod/deep-ml/discussions)

## ⭐ Star History

If you find this project useful, please consider giving it a star!

## Citation

If you use deep-ml in your research, please cite:

```bibtex
@software{deepml2026,
  author = {Rathod, Sagar},
  title = {deep-ml: PyTorch Training Framework},
  year = {2026},
  url = {https://github.com/sagar100rathod/deep-ml}
}
```
