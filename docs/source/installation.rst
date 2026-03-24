Installation
============

Requirements
------------

- Python 3.8 or higher
- PyTorch 1.12 or higher
- CUDA (optional, for GPU support)

Basic Installation
------------------

Install from PyPI using pip:

.. code-block:: bash

   pip install deepml

With GPU Support
----------------

Before installing deep-ml, install PyTorch with CUDA support. Visit the
`PyTorch installation page <https://pytorch.org/get-started/locally/>`_ to get
the appropriate command for your system.

For example, with CUDA 11.8:

.. code-block:: bash

   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   pip install deepml

Development Installation
------------------------

To install from source for development:

.. code-block:: bash

   git clone https://github.com/sagar100rathod/deep-ml.git
   cd deep-ml
   pip install -e ".[dev]"

Optional Dependencies
---------------------

Distributed Training
~~~~~~~~~~~~~~~~~~~~

For Lightning Fabric support:

.. code-block:: bash

   pip install lightning-fabric

For HuggingFace Accelerate support:

.. code-block:: bash

   pip install accelerate

Experiment Tracking
~~~~~~~~~~~~~~~~~~~

For MLflow:

.. code-block:: bash

   pip install mlflow

For Weights & Biases:

.. code-block:: bash

   pip install wandb

Data Augmentation
~~~~~~~~~~~~~~~~~

For advanced augmentations with Albumentations:

.. code-block:: bash

   pip install albumentations

Verifying Installation
----------------------

To verify your installation:

.. code-block:: python

   import deepml
   print(deepml.__version__)

   # Check available trainers
   from deepml.fabric_trainer import FabricTrainer
   from deepml.accelerator_trainer import AcceleratorTrainer
   print("Installation successful!")
