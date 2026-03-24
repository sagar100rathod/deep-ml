Quick Start Guide
=================

This guide will walk you through creating your first training pipeline with deep-ml.

Image Classification Example
-----------------------------

Let's train a ResNet-18 model on CIFAR-10:

Step 1: Prepare Data
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from torch.utils.data import DataLoader
   from torchvision import datasets, transforms

   # Define transformations
   transform = transforms.Compose([
       transforms.ToTensor(),
       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
   ])

   # Load datasets
   train_dataset = datasets.CIFAR10(
       root='./data',
       train=True,
       download=True,
       transform=transform
   )

   val_dataset = datasets.CIFAR10(
       root='./data',
       train=False,
       download=True,
       transform=transform
   )

   # Create dataloaders
   train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
   val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

Step 2: Define Model and Task
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from torchvision.models import resnet18
   from deepml.tasks import ImageClassification

   # Create model
   model = resnet18(num_classes=10)

   # Create task
   task = ImageClassification(
       model=model,
       model_dir='./checkpoints',
       classes=['airplane', 'automobile', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck']
   )

Step 3: Setup Optimizer and Loss
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch
   from torch.optim import Adam

   optimizer = Adam(model.parameters(), lr=1e-3)
   criterion = torch.nn.CrossEntropyLoss()

Step 4: Create Trainer
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from deepml.fabric_trainer import FabricTrainer

   trainer = FabricTrainer(
       task=task,
       optimizer=optimizer,
       criterion=criterion,
       accelerator='auto',  # Use GPU if available
       devices='auto',      # Use all available devices
       precision='16-mixed' # Mixed precision training
   )

Step 5: Train the Model
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from deepml.metrics.classification import Accuracy

   # Define metrics
   metrics = {
       'accuracy': Accuracy()
   }

   # Start training
   trainer.fit(
       train_loader=train_loader,
       val_loader=val_loader,
       epochs=50,
       metrics=metrics,
       save_model_after_every_epoch=10
   )

Step 6: Make Predictions
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Visualize predictions
   trainer.show_predictions(
       loader=val_loader,
       samples=9,
       cols=3
   )

   # Get predictions
   predictions, targets = task.predict(val_loader)

Semantic Segmentation Example
------------------------------

Train a U-Net for binary segmentation:

.. code-block:: python

   from deepml.tasks import Segmentation
   from deepml.fabric_trainer import FabricTrainer
   import torch

   # Define model (you need to implement or import UNet)
   model = UNet(in_channels=3, out_channels=1)

   # Create task
   task = Segmentation(
       model=model,
       model_dir='./checkpoints',
       mode='binary',
       num_classes=1,
       threshold=0.5
   )

   # Setup training
   optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
   criterion = torch.nn.BCEWithLogitsLoss()

   trainer = FabricTrainer(
       task=task,
       optimizer=optimizer,
       criterion=criterion
   )

   # Train
   trainer.fit(
       train_loader=train_loader,
       val_loader=val_loader,
       epochs=100
   )

Using Accelerate Trainer
-------------------------

For HuggingFace Accelerate integration:

.. code-block:: python

   from deepml.accelerator_trainer import AcceleratorTrainer

   trainer = AcceleratorTrainer(
       task=task,
       optimizer=optimizer,
       criterion=criterion,
       accelerator_config={
           'gradient_accumulation_steps': 4,
           'mixed_precision': 'fp16'
       }
   )

   trainer.fit(
       train_loader=train_loader,
       val_loader=val_loader,
       epochs=50,
       gradient_clip_max_norm=1.0
   )

Learning Rate Scheduling
-------------------------

Use OneCycleLR with warmup:

.. code-block:: python

   from deepml.lr_scheduler_utils import setup_one_cycle_lr_scheduler_with_warmup

   lr_scheduler = setup_one_cycle_lr_scheduler_with_warmup(
       optimizer=optimizer,
       steps_per_epoch=len(train_loader),
       warmup_ratio=0.1,
       num_epochs=50,
       max_lr=1e-3
   )

   trainer = FabricTrainer(
       task=task,
       optimizer=optimizer,
       criterion=criterion,
       lr_scheduler_fn=lambda opt: lr_scheduler
   )

Experiment Tracking
-------------------

Track experiments with MLflow:

.. code-block:: python

   from deepml.tracking import MLFlowLogger

   logger = MLFlowLogger(
       experiment_name='cifar10-classification',
       tracking_uri='./mlruns'
   )

   trainer.fit(
       train_loader=train_loader,
       val_loader=val_loader,
       epochs=50,
       logger=logger
   )

Next Steps
----------

- Explore the :doc:`trainers` guide for advanced training options
- Learn about :doc:`tasks` for different problem types
- Check out :doc:`examples` for more use cases
