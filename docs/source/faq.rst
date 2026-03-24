Frequently Asked Questions
==========================

General Questions
-----------------

What is deep-ml?
~~~~~~~~~~~~~~~~

deep-ml is a high-level PyTorch training framework that simplifies deep learning workflows
for computer vision tasks. It provides ready-to-use trainers, task implementations, and
experiment tracking.

Why use deep-ml instead of pure PyTorch?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Less boilerplate**: No need to write training loops, device management, distributed training setup
- **Best practices**: Gradient accumulation, clipping, mixed precision built-in
- **Experiment tracking**: Seamless TensorBoard, MLflow, and wandb integration
- **Reproducibility**: Consistent checkpoint management and state restoration

How does deep-ml compare to PyTorch Lightning?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

deep-ml is simpler and more focused:

- Smaller API surface
- Focused on computer vision
- Multiple backend options (Fabric, Accelerate)
- Less opinionated about code structure

PyTorch Lightning is more comprehensive but has a steeper learning curve.

Installation & Setup
--------------------

What Python version is required?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Python 3.8 or higher is required.

Do I need CUDA for deep-ml?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

No, deep-ml works on CPU, but training will be slower. For GPU training, install
PyTorch with CUDA support before installing deep-ml.

Can I use deep-ml with Apple Silicon (M1/M2)?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Yes! Use ``device='mps'`` or ``accelerator='mps'`` to leverage Apple Silicon GPU:

.. code-block:: python

   trainer = FabricTrainer(
       ...,
       accelerator='mps'
   )

Training
--------

How do I resume training from a checkpoint?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   trainer.fit(
       ...,
       resume_from_checkpoint='./checkpoints/epoch_50_model.pt',
       load_optimizer_state=True,
       load_scheduler_state=True
   )

How do I use mixed precision training?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # FabricTrainer
   trainer = FabricTrainer(
       ...,
       precision='16-mixed'  # or 'bf16-mixed'
   )

   # AcceleratorTrainer
   trainer = AcceleratorTrainer(
       ...,
       accelerator_config={'mixed_precision': 'fp16'}
   )

How do I implement gradient accumulation?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   trainer.fit(
       ...,
       gradient_accumulation_steps=4  # Effective batch = 4 * batch_size
   )

My training is slow. How can I speed it up?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Use mixed precision**: ``precision='16-mixed'``
2. **Increase batch size**: Larger batches = fewer iterations
3. **Use multiple workers**: ``num_workers=4`` in DataLoader
4. **Enable pin_memory**: ``pin_memory=True`` in DataLoader
5. **Use gradient accumulation**: Instead of increasing batch size
6. **Profile your code**: Identify bottlenecks

.. code-block:: python

   # Fast DataLoader configuration
   loader = DataLoader(
       dataset,
       batch_size=64,  # As large as GPU memory allows
       shuffle=True,
       num_workers=4,  # Parallel data loading
       pin_memory=True,  # Faster GPU transfer
       persistent_workers=True  # Keep workers alive
   )

How do I handle class imbalance?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Option 1: Weighted loss
   from deepml.losses import WeightedBCEWithLogitsLoss

   criterion = WeightedBCEWithLogitsLoss(w_p=10.0, w_n=1.0)

   # Option 2: Class weights
   class_weights = torch.tensor([1.0, 10.0])
   criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

   # Option 3: Weighted sampling
   from torch.utils.data import WeightedRandomSampler

   samples_weight = compute_sample_weights(train_dataset)
   sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
   train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)

Distributed Training
--------------------

How do I train on multiple GPUs?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   trainer = FabricTrainer(
       ...,
       accelerator='gpu',
       strategy='ddp',  # DistributedDataParallel
       devices='auto'  # Use all available GPUs
   )

Can I train across multiple machines?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Yes, with FabricTrainer:

.. code-block:: python

   trainer = FabricTrainer(
       ...,
       accelerator='gpu',
       strategy='ddp',
       devices=4,  # GPUs per node
       num_nodes=2  # Number of machines
   )

Run on each node:

.. code-block:: bash

   # Node 0 (master)
   fabric run --node-rank=0 --num-nodes=2 --main-address=192.168.1.1 train.py

   # Node 1
   fabric run --node-rank=1 --num-nodes=2 --main-address=192.168.1.1 train.py

What's the difference between DP and DDP?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **DP (DataParallel)**: Single process, thread-based, slower, easier to debug
- **DDP (DistributedDataParallel)**: Multi-process, faster, recommended

Use DDP for training, DP for quick debugging.

Data & Datasets
---------------

How do I use custom datasets?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Inherit from ``torch.utils.data.Dataset``:

.. code-block:: python

   class MyDataset(torch.utils.data.Dataset):
       def __init__(self, ...):
           # Load data
           pass

       def __len__(self):
           return num_samples

       def __getitem__(self, idx):
           # Load and return (image, label)
           return image, label

   dataset = MyDataset(...)
   loader = DataLoader(dataset, batch_size=32)

How do I handle large datasets?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Use streaming**: Load data on-the-fly, don't load all into memory
2. **Use multiple workers**: Parallel data loading
3. **Use pin_memory**: Faster GPU transfers
4. **Consider data format**: Use efficient formats (LMDB, HDF5, WebDataset)

.. code-block:: python

   from torch.utils.data import IterableDataset

   class StreamingDataset(IterableDataset):
       def __iter__(self):
           # Stream data from disk/network
           for sample in data_source:
               yield preprocess(sample)

How do I apply data augmentation?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use torchvision transforms or Albumentations:

.. code-block:: python

   # torchvision (for classification)
   from torchvision import transforms

   transform = transforms.Compose([
       transforms.RandomResizedCrop(224),
       transforms.RandomHorizontalFlip(),
       transforms.ToTensor()
   ])

   # Albumentations (for segmentation)
   import albumentations as A
   from albumentations.pytorch import ToTensorV2

   transform = A.Compose([
       A.Resize(512, 512),
       A.HorizontalFlip(p=0.5),
       A.Normalize(),
       ToTensorV2()
   ])

Models
------

Can I use any PyTorch model?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Yes! deep-ml works with any ``torch.nn.Module``:

.. code-block:: python

   # torchvision models
   from torchvision.models import resnet50

   # timm models
   import timm
   model = timm.create_model('efficientnet_b0', pretrained=True)

   # Custom models
   class MyModel(torch.nn.Module):
       ...

   # All work with deep-ml
   task = ImageClassification(model=model, ...)

How do I use a pre-trained model?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from torchvision.models import resnet50, ResNet50_Weights

   model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

   # Replace classifier for your number of classes
   model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

How do I freeze layers?
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Freeze all layers
   for param in model.parameters():
       param.requires_grad = False

   # Unfreeze specific layers
   for param in model.fc.parameters():
       param.requires_grad = True

   # Create optimizer only for trainable parameters
   optimizer = torch.optim.Adam(
       filter(lambda p: p.requires_grad, model.parameters()),
       lr=1e-3
   )

Errors & Debugging
------------------

CUDA out of memory error
~~~~~~~~~~~~~~~~~~~~~~~~

Solutions:

1. **Reduce batch size**
2. **Use gradient accumulation**:

   .. code-block:: python

      trainer.fit(..., gradient_accumulation_steps=4)

3. **Use mixed precision**:

   .. code-block:: python

      trainer = FabricTrainer(..., precision='16-mixed')

4. **Clear cache**:

   .. code-block:: python

      torch.cuda.empty_cache()

5. **Check for memory leaks**: Don't accumulate tensors in lists during training

Validation loss is NaN
~~~~~~~~~~~~~~~~~~~~~~

Possible causes:

1. **Learning rate too high**: Reduce it
2. **Gradient explosion**: Use gradient clipping
3. **Numerical instability**: Use ``BCEWithLogitsLoss`` instead of ``BCELoss``
4. **Invalid inputs**: Check for NaN/Inf in data

.. code-block:: python

   # Add gradient clipping
   trainer.fit(
       ...,
       gradient_clip_max_norm=1.0
   )

   # Check data
   for x, y in train_loader:
       assert not torch.isnan(x).any()
       assert not torch.isinf(x).any()
       break

Model not learning (loss not decreasing)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Check:

1. **Learning rate**: Try different values (1e-4 to 1e-2)
2. **Model frozen**: Ensure layers are trainable
3. **Loss function**: Correct for your task?
4. **Data preprocessing**: Normalized correctly?
5. **Batch size**: Not too small?

.. code-block:: python

   # Debug: check gradient flow
   for name, param in model.named_parameters():
       if param.grad is not None:
           print(f"{name}: {param.grad.abs().mean()}")

Performance
-----------

How many epochs should I train?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Depends on:

- Dataset size: Smaller → more epochs needed
- Model complexity: Larger → more epochs
- Learning rate: Lower → more epochs

General guidelines:

- Small datasets (<10K images): 100-500 epochs
- Medium datasets (10K-100K): 50-100 epochs
- Large datasets (>100K): 20-50 epochs

Use early stopping:

.. code-block:: python

   patience = 10
   best_loss = float('inf')
   epochs_without_improvement = 0

   for epoch in range(max_epochs):
       trainer.fit(..., epochs=1)

       if trainer.best_val_loss < best_loss:
           best_loss = trainer.best_val_loss
           epochs_without_improvement = 0
       else:
           epochs_without_improvement += 1

       if epochs_without_improvement >= patience:
           print("Early stopping!")
           break

What learning rate should I use?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use learning rate finder:

.. code-block:: python

   from torch_lr_finder import LRFinder

   lr_finder = LRFinder(model, optimizer, criterion)
   lr_finder.range_test(train_loader, end_lr=1, num_iter=100)
   lr_finder.plot()
   lr_finder.reset()

General guidelines:

- Adam/AdamW: 1e-3 to 1e-4
- SGD: 1e-1 to 1e-2
- Fine-tuning: 1e-4 to 1e-5

Compatibility
-------------

What PyTorch version is required?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PyTorch 1.12 or higher. Latest version recommended.

Does deep-ml work with torch.compile?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Yes! Available in PyTorch 2.0+:

.. code-block:: python

   model = torch.compile(model)

   task = ImageClassification(model=model, ...)
   trainer.fit(...)

Can I use deep-ml with other libraries?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Yes:

- **timm**: ``import timm; model = timm.create_model(...)``
- **transformers**: Works with Vision Transformers
- **segmentation-models-pytorch**: Pre-built segmentation models
- **albumentations**: For data augmentation
- **torchmetrics**: Additional metrics

Getting Help
------------

Where can I get help?
~~~~~~~~~~~~~~~~~~~~~

1. Check this documentation
2. Review examples in the repository
3. Open an issue on GitHub
4. Check existing issues for solutions

How do I report a bug?
~~~~~~~~~~~~~~~~~~~~~~~

Open an issue on GitHub with:

1. Minimal reproducible example
2. Error message and stack trace
3. Environment info (Python, PyTorch, deep-ml versions)
4. Expected vs actual behavior

How do I request a feature?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Open a feature request on GitHub describing:

1. Use case
2. Proposed API
3. Why it's useful
4. Are you willing to contribute?

Contributing
------------

See :doc:`contributing` for guidelines on contributing to deep-ml.
