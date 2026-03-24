Trainers
========

deep-ml provides three trainer implementations for different use cases.

FabricTrainer
-------------

**Recommended** for most use cases. Uses Lightning Fabric for seamless distributed training.

Features
~~~~~~~~

- Distributed training (DDP, FSDP, DeepSpeed)
- Mixed precision training
- Multi-GPU support
- Automatic device placement
- Simple API

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from deepml.fabric_trainer import FabricTrainer

   trainer = FabricTrainer(
       task=task,
       optimizer=optimizer,
       criterion=criterion,
       accelerator='auto',     # 'cpu', 'cuda', 'mps', 'gpu', 'tpu'
       strategy='auto',        # 'dp', 'ddp', 'fsdp', 'deepspeed'
       devices='auto',         # Number of devices or 'auto'
       precision='32-true',    # '16-mixed', '32-true', 'bf16-mixed'
       num_nodes=1             # For multi-node training
   )

Configuration Options
~~~~~~~~~~~~~~~~~~~~~

**Accelerator Options:**

- ``'cpu'``: CPU training
- ``'cuda'`` / ``'gpu'``: Single or multi-GPU
- ``'mps'``: Apple Silicon GPU
- ``'tpu'``: Google Cloud TPU
- ``'auto'``: Automatic selection

**Strategy Options:**

- ``'dp'``: DataParallel (single-node)
- ``'ddp'``: DistributedDataParallel (recommended)
- ``'fsdp'``: Fully Sharded Data Parallel
- ``'deepspeed'``: Microsoft DeepSpeed
- ``'auto'``: Automatic selection

**Precision Options:**

- ``'32-true'``: Full precision (FP32)
- ``'16-mixed'``: Mixed precision (FP16)
- ``'bf16-mixed'``: Mixed precision (BF16)
- ``'64-true'``: Double precision (FP64)

Training Method
~~~~~~~~~~~~~~~

.. code-block:: python

   trainer.fit(
       train_loader=train_loader,
       val_loader=val_loader,
       epochs=50,
       save_model_after_every_epoch=10,
       metrics={'accuracy': Accuracy()},
       gradient_accumulation_steps=4,
       gradient_clip_value=1.0,          # Clip by value
       gradient_clip_max_norm=None,      # Clip by norm
       resume_from_checkpoint='path/to/checkpoint.pt',
       load_optimizer_state=True,
       load_scheduler_state=True,
       logger=mlflow_logger,
       non_blocking=True,
       image_inverse_transform=denormalize,
       logger_img_size=224
   )

Advanced: Multi-Node Training
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Node 0
   fabric run --node-rank=0 --num-nodes=2 --main-address=192.168.1.1 train.py

   # Node 1
   fabric run --node-rank=1 --num-nodes=2 --main-address=192.168.1.1 train.py

AcceleratorTrainer
------------------

Uses HuggingFace Accelerate for distributed training with additional flexibility.

Features
~~~~~~~~

- Same distributed strategies as FabricTrainer
- Compatible with Accelerate CLI
- Fine-grained control over gradient synchronization
- Easy integration with Transformers

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from deepml.accelerator_trainer import AcceleratorTrainer

   trainer = AcceleratorTrainer(
       task=task,
       optimizer=optimizer,
       criterion=criterion,
       lr_scheduler=lr_scheduler,  # Note: instance, not factory
       lr_scheduler_step_policy='epoch',
       accelerator_config={
           'gradient_accumulation_steps': 4,
           'mixed_precision': 'fp16',
           'device_placement': True,
           'split_batches': False
       }
   )

Accelerator Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   accelerator_config = {
       # Gradient accumulation
       'gradient_accumulation_steps': 4,

       # Mixed precision
       'mixed_precision': 'fp16',  # 'no', 'fp16', 'bf16'

       # Device settings
       'device_placement': True,
       'split_batches': False,

       # Logging
       'log_with': 'tensorboard',
       'project_dir': './logs',

       # Advanced
       'dispatch_batches': None,
       'even_batches': True,
       'step_scheduler_with_optimizer': True
   }

Using Accelerate CLI
~~~~~~~~~~~~~~~~~~~~

Create ``accelerate_config.yaml``:

.. code-block:: yaml

   compute_environment: LOCAL_MACHINE
   distributed_type: MULTI_GPU
   mixed_precision: fp16
   num_processes: 4
   gpu_ids: all

Run training:

.. code-block:: bash

   accelerate launch --config_file accelerate_config.yaml train.py

Learner (Deprecated)
--------------------

.. warning::
   This trainer is deprecated. Use FabricTrainer or AcceleratorTrainer instead.

Classic PyTorch trainer with manual device management.

.. code-block:: python

   from deepml.trainer import Learner

   learner = Learner(
       task=task,
       optimizer=optimizer,
       criterion=criterion,
       lr_scheduler=lr_scheduler,
       use_amp=True  # Automatic Mixed Precision
   )

   learner.fit(
       train_loader=train_loader,
       val_loader=val_loader,
       epochs=50
   )

Choosing a Trainer
------------------

**Use FabricTrainer when:**

- You want the easiest distributed training setup
- You're starting a new project
- You need multi-node training
- You want Lightning ecosystem integration

**Use AcceleratorTrainer when:**

- You're using HuggingFace models/ecosy stem
- You need fine-grained control over distributed training
- You prefer the Accelerate CLI workflow
- You're migrating from existing Accelerate code

**Don't use Learner:**

- It's deprecated and will be removed in future versions
- Use FabricTrainer or AcceleratorTrainer for new projects

Common Training Patterns
------------------------

Gradient Accumulation
~~~~~~~~~~~~~~~~~~~~~

Simulate larger batch sizes:

.. code-block:: python

   trainer.fit(
       train_loader=train_loader,
       val_loader=val_loader,
       epochs=50,
       gradient_accumulation_steps=8  # Effective batch size = 8 * batch_size
   )

Gradient Clipping
~~~~~~~~~~~~~~~~~

Prevent exploding gradients:

.. code-block:: python

   # Clip by value
   trainer.fit(
       ...,
       gradient_clip_value=1.0
   )

   # Clip by norm (recommended)
   trainer.fit(
       ...,
       gradient_clip_max_norm=1.0
   )

Learning Rate Scheduling
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from torch.optim.lr_scheduler import CosineAnnealingLR

   # For FabricTrainer: use factory function
   lr_scheduler_fn = lambda opt: CosineAnnealingLR(opt, T_max=50)

   trainer = FabricTrainer(
       ...,
       lr_scheduler_fn=lr_scheduler_fn,
       lr_scheduler_step_policy='epoch'  # or 'step'
   )

   # For AcceleratorTrainer: use instance
   lr_scheduler = CosineAnnealingLR(optimizer, T_max=50)

   trainer = AcceleratorTrainer(
       ...,
       lr_scheduler=lr_scheduler,
       lr_scheduler_step_policy='epoch'
   )

Resume Training
~~~~~~~~~~~~~~~

.. code-block:: python

   trainer.fit(
       ...,
       resume_from_checkpoint='./checkpoints/best_val_model.pt',
       load_optimizer_state=True,
       load_scheduler_state=True
   )

Checkpoint Management
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   trainer.fit(
       ...,
       save_model_after_every_epoch=10  # Save every 10 epochs
   )

   # Checkpoints saved:
   # - best_val_model.pt (best validation loss)
   # - epoch_10_model.pt, epoch_20_model.pt, ...
   # - latest_model.pt (most recent)
