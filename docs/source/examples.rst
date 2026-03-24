Examples
========

Complete examples for various use cases.

Image Classification
--------------------

CIFAR-10 with ResNet
~~~~~~~~~~~~~~~~~~~~

See :doc:`tutorials` for a complete CIFAR-10 classification example.

Transfer Learning
~~~~~~~~~~~~~~~~~

Fine-tune a pre-trained ResNet50 on a custom dataset:

.. code-block:: bash

   cd scripts
   python torch_trainer_mnist_example.py

Semantic Segmentation
---------------------

Binary Segmentation
~~~~~~~~~~~~~~~~~~~

Road segmentation example:

.. code-block:: bash

   cd notebooks
   jupyter notebook Road_Segmentation_Example.ipynb

Multiclass Segmentation
~~~~~~~~~~~~~~~~~~~~~~~

Scene segmentation with multiple classes:

.. code-block:: bash

   cd notebooks
   jupyter notebook Multiclass_Scene_Segmentation.ipynb

Or run the script:

.. code-block:: bash

   cd scripts
   python fabric_trainer_multiclass_segmentation_example.py

Image Regression
----------------

Age/Depth Estimation
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   cd notebooks
   jupyter notebook Image_Regression_Example.ipynb

Distributed Training
--------------------

FabricTrainer Example
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   cd scripts
   python fabric_trainer_mnist_example.py

AcceleratorTrainer Example
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   cd scripts
   python accelerator_trainer_mnist_example.py

Multi-GPU Training
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Automatically uses all available GPUs
   python torch_trainer_binary_segmentation_example.py

Experiment Tracking
-------------------

MLflow Integration
~~~~~~~~~~~~~~~~~~

Check the :doc:`tracking` documentation for MLflow examples.

Weights & Biases
~~~~~~~~~~~~~~~~

Examples with wandb integration are available in the scripts directory.

Custom Metrics
--------------

Implementing Custom Metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

See :doc:`metrics` for examples of custom metric implementations.

Project Structure
-----------------

Recommended Structure
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   my_project/
   ├── data/
   │   ├── train/
   │   ├── val/
   │   └── test/
   ├── checkpoints/
   ├── runs/  # TensorBoard logs
   ├── mlruns/  # MLflow logs
   ├── notebooks/
   │   └── exploration.ipynb
   ├── src/
   │   ├── models/
   │   │   ├── __init__.py
   │   │   └── custom_model.py
   │   ├── datasets/
   │   │   ├── __init__.py
   │   │   └── custom_dataset.py
   │   └── utils/
   │       ├── __init__.py
   │       └── helpers.py
   ├── train.py
   ├── evaluate.py
   ├── predict.py
   └── requirements.txt

Example train.py
~~~~~~~~~~~~~~~~

.. code-block:: python

   import argparse
   import torch
   from torch.utils.data import DataLoader
   from deepml.tasks import ImageClassification
   from deepml.fabric_trainer import FabricTrainer

   def main(args):
       # Data
       train_dataset = create_dataset(args.data_dir, split='train')
       val_dataset = create_dataset(args.data_dir, split='val')

       train_loader = DataLoader(
           train_dataset,
           batch_size=args.batch_size,
           shuffle=True,
           num_workers=args.num_workers
       )

       val_loader = DataLoader(
           val_dataset,
           batch_size=args.batch_size,
           num_workers=args.num_workers
       )

       # Model
       model = create_model(
           arch=args.arch,
           num_classes=args.num_classes,
           pretrained=args.pretrained
       )

       # Task
       task = ImageClassification(
           model=model,
           model_dir=args.checkpoint_dir
       )

       # Training setup
       optimizer = torch.optim.Adam(
           model.parameters(),
           lr=args.lr,
           weight_decay=args.weight_decay
       )

       criterion = torch.nn.CrossEntropyLoss()

       # Trainer
       trainer = FabricTrainer(
           task=task,
           optimizer=optimizer,
           criterion=criterion,
           accelerator=args.accelerator,
           devices=args.devices,
           precision=args.precision
       )

       # Train
       trainer.fit(
           train_loader=train_loader,
           val_loader=val_loader,
           epochs=args.epochs,
           save_model_after_every_epoch=args.save_freq
       )

   if __name__ == '__main__':
       parser = argparse.ArgumentParser()
       parser.add_argument('--data_dir', type=str, required=True)
       parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
       parser.add_argument('--arch', type=str, default='resnet18')
       parser.add_argument('--num_classes', type=int, required=True)
       parser.add_argument('--pretrained', action='store_true')
       parser.add_argument('--batch_size', type=int, default=32)
       parser.add_argument('--epochs', type=int, default=50)
       parser.add_argument('--lr', type=float, default=1e-3)
       parser.add_argument('--weight_decay', type=float, default=1e-4)
       parser.add_argument('--num_workers', type=int, default=4)
       parser.add_argument('--accelerator', type=str, default='auto')
       parser.add_argument('--devices', type=str, default='auto')
       parser.add_argument('--precision', type=str, default='16-mixed')
       parser.add_argument('--save_freq', type=int, default=10)

       args = parser.parse_args()
       main(args)

Run Training
~~~~~~~~~~~~

.. code-block:: bash

   python train.py \
       --data_dir ./data \
       --checkpoint_dir ./checkpoints \
       --arch resnet50 \
       --num_classes 10 \
       --pretrained \
       --batch_size 64 \
       --epochs 100 \
       --lr 1e-3

More Examples
-------------

Check the following for more examples:

- ``scripts/`` directory in the repository
- ``notebooks/`` directory for Jupyter notebooks
- GitHub repository: https://github.com/sagar100rathod/deep-ml

Community Examples
------------------

Share your examples and projects using deep-ml!

Contributing examples is welcome. See :doc:`contributing` for guidelines.
