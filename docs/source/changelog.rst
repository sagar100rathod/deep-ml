Changelog
=========

Version 3.3.0
-------------

**Breaking Changes:**

- Classification metrics (``Accuracy``, ``Precision``, ``Recall``, ``FScore``,
  ``MCC``) now require ``num_classes=N`` for multiclass ``(N, C)`` inputs.
  Without it, binary mode is assumed and a clear ``ValueError`` is raised on 2D input.
- Classification ``epsilon`` parameter replaced by ``zero_division`` (default ``0.0``).
- ``Binarizer`` class removed from ``deepml.metrics.classification``.
- ``deepml.metrics.commons`` removed from the public API (internal utility).
- Segmentation metric default ``reduction`` changed from ``'macro-imagewise'`` to
  ``'macro'``. Imagewise variants (``'macro-imagewise'``, ``'micro-imagewise'``,
  ``'weighted-imagewise'``) are no longer supported.
- Old segmentation classes ``IoU``, ``DiceCoefficient``, ``PixelAccuracy`` replaced
  by ``IoUScore``, ``F1Score``, ``Accuracy``.

**New Features:**

- **Stateful metric accumulation**: all built-in metrics now accumulate raw
  TP/FP/FN/TN counts across the full epoch via torchmetrics and compute the metric
  value once at epoch end. This fixes the Jensen bias in per-batch SMA averaging —
  ``mean(per_batch_IoU) ≠ global_IoU``. The progress bar shows the running
  epoch-to-date value.
- ``steps_per_epoch`` parameter added to ``FabricTrainer.fit()`` and
  ``AcceleratorTrainer.fit()``. Supports streaming/IterableDatasets (no ``__len__``)
  and synthetic epoch boundaries over very large fixed datasets.
- Stateful metrics are automatically moved to the training device (CUDA/MPS) at the
  start of ``fit()``. No manual ``.to(device)`` call is needed.

**New Dependency:**

- ``torchmetrics>=0.11.0`` is now a required core dependency.

----

Version 0.3.0 (Upcoming)
------------------------

**New Features:**

- Added Google-style docstrings to all modules
- Enhanced documentation with comprehensive guides
- Improved error messages and validation

**Bug Fixes:**

- Fixed assertion logic in ``lr_scheduler_utils.py`` for warmup validation
- Fixed gradient clipping synchronization in ``AcceleratorTrainer``
- Fixed off-by-one in ``FabricTrainer`` gradient accumulation. The optimizer
  stepped when ``batch_index % gradient_accumulation_steps == 0``, so it fired
  on the *first* micro-batch of every epoch (applying a gradient scaled by
  ``1 / gradient_accumulation_steps``) and produced one to two extra steps per
  epoch. Schedulers using ``lr_scheduler_step_policy="step"`` and sized from
  ``steps_per_epoch`` therefore overran ``total_steps`` and raised
  ``ValueError`` late in long runs. The count is now exactly
  ``ceil(num_batches / gradient_accumulation_steps)``, matching ``Learner``.
  See :ref:`steps-per-epoch` for how to size a schedule correctly.

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
