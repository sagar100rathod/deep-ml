Changelog
=========

Version 0.3.0 (Upcoming)
------------------------

**New Features:**

- Added Google-style docstrings to all modules
- Enhanced documentation with comprehensive guides
- Improved error messages and validation

**Bug Fixes:**

- Fixed assertion logic in ``lr_scheduler_utils.py`` for warmup validation
- Fixed gradient clipping synchronization in ``AcceleratorTrainer``

**Improvements:**

- Better type hints throughout the codebase
- Comprehensive test coverage
- Improved examples and tutorials

**Deprecations:**

- ``Learner`` class is now deprecated, use ``FabricTrainer`` or ``AcceleratorTrainer``

Version 0.2.0
-------------

**New Features:**

- Added ``AcceleratorTrainer`` for HuggingFace Accelerate support
- Added ``FabricTrainer`` for Lightning Fabric support
- Support for multi-label image classification
- Added experiment tracking (MLflow, wandb)
- Learning rate scheduler utilities with warmup

**Improvements:**

- Better distributed training support
- Improved checkpoint management
- Enhanced visualization tools

Version 0.1.0
-------------

**Initial Release:**

- Basic ``Learner`` trainer implementation
- Image classification support
- Semantic segmentation support
- Image regression support
- TensorBoard integration
- Basic metrics (Accuracy, IoU, Dice)
- Custom loss functions (Jaccard, RMSE, Contrastive, Angular)

Migration Guide
===============

Migrating from Learner to FabricTrainer
----------------------------------------

Old Code
~~~~~~~~

.. code-block:: python

   from deepml.trainer import Learner

   learner = Learner(
       task=task,
       optimizer=optimizer,
       criterion=criterion,
       lr_scheduler=lr_scheduler,
       use_amp=True
   )

   learner.fit(
       train_loader=train_loader,
       val_loader=val_loader,
       epochs=50
   )

New Code
~~~~~~~~

.. code-block:: python

   from deepml.fabric_trainer import FabricTrainer

   # Note: lr_scheduler_fn instead of lr_scheduler
   lr_scheduler_fn = lambda opt: CosineAnnealingLR(opt, T_max=50)

   trainer = FabricTrainer(
       task=task,
       optimizer=optimizer,
       criterion=criterion,
       lr_scheduler_fn=lr_scheduler_fn,
       precision='16-mixed'  # Instead of use_amp=True
   )

   trainer.fit(
       train_loader=train_loader,
       val_loader=val_loader,
       epochs=50
   )

Key Differences
~~~~~~~~~~~~~~~

1. **lr_scheduler**: Instance → Factory function
2. **use_amp**: Boolean → ``precision`` parameter
3. **Device management**: Manual → Automatic
4. **Distributed training**: Manual setup → Automatic

Breaking Changes
----------------

Version 0.3.0
~~~~~~~~~~~~~

- None (backward compatible)

Version 0.2.0
~~~~~~~~~~~~~

- Changed import paths for some utilities
- Modified Task API signatures
- Updated checkpoint format (backward compatible loading)

Future Plans
------------

Version 0.4.0 (Planned)
~~~~~~~~~~~~~~~~~~~~~~~

- Remove deprecated ``Learner`` class
- Add support for object detection tasks
- Enhanced callback system
- Better gradient accumulation handling
- Support for DDP with model sharding

Version 0.5.0 (Planned)
~~~~~~~~~~~~~~~~~~~~~~~

- Multi-task learning support
- Advanced augmentation strategies
- Model ensemble utilities
- Automatic hyperparameter tuning integration
- Production deployment utilities
