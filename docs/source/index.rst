.. deep-ml documentation master file

Welcome to deep-ml's documentation!
====================================

.. image:: https://img.shields.io/badge/License-MIT-green
   :target: https://opensource.org/licenses/MIT
   :alt: License: MIT

.. image:: https://img.shields.io/badge/Python-3.11%2B-orange
   :alt: Python Version

.. image:: https://readthedocs.org/projects/deep-ml/badge/?version=latest
   :target: https://deep-ml.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

.. image:: https://static.pepy.tech/personalized-badge/deepml?period=total&units=international_system&left_color=black&right_color=brightgreen&left_text=Downloads
   :target: https://pepy.tech/project/deepml
   :alt: Downloads

**deep-ml** is a high-level PyTorch training framework that simplifies deep learning workflows
for computer vision tasks. It provides easy-to-use trainers with distributed training support,
comprehensive task implementations, and seamless experiment tracking.

Key Features
------------

🚀 **Multiple Training Backends**
   - **FabricTrainer**: Lightning Fabric for distributed training (recommended for multi-GPU)
   - **AcceleratorTrainer**: HuggingFace Accelerate integration (recommended for multi-GPU)
   - **Learner**: Classic PyTorch trainer (single-device, notebook-friendly)

🎯 **Pre-built Task Implementations**
   - Image Classification (single & multi-label)
   - Semantic Segmentation (binary & multiclass)
   - Image Regression
   - Custom tasks via extensible base classes

📊 **Experiment Tracking**
   - TensorBoard integration
   - MLflow support
   - Weights & Biases (wandb) integration
   - Custom logger interface

🔧 **Advanced Training Features**
   - Automatic Mixed Precision (AMP)
   - Gradient accumulation & clipping
   - Learning rate scheduling with warmup
   - Multi-GPU and distributed training
   - Checkpoint management

Quick Start
-----------

Installation
~~~~~~~~~~~~

.. code-block:: bash

   pip install deepml

Basic Usage
~~~~~~~~~~~

.. code-block:: python

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
       model_dir="./checkpoints"
   )

   # 3. Setup optimizer and loss
   optimizer = Adam(model.parameters(), lr=1e-3)
   criterion = torch.nn.CrossEntropyLoss()

   # 4. Create trainer
   trainer = FabricTrainer(
       task=task,
       optimizer=optimizer,
       criterion=criterion,
       accelerator="auto",
       devices="auto"
   )

   # 5. Train!
   trainer.fit(
       train_loader=train_loader,
       val_loader=val_loader,
       epochs=50
   )

Table of Contents
-----------------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart
   tutorials

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   trainers
   tasks
   datasets
   losses
   metrics
   tracking
   visualization

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   modules
   api/deepml

.. toctree::
   :maxdepth: 1
   :caption: Additional Resources

   examples
   faq
   changelog
   contributing

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
