Experiment Tracking
===================

deep-ml supports multiple experiment tracking platforms.

TensorBoard Logger
------------------

Default logger that integrates with TensorBoard.

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from deepml.tracking import TensorboardLogger

   logger = TensorboardLogger(model_dir='./runs/experiment_1')

   trainer.fit(
       train_loader=train_loader,
       val_loader=val_loader,
       epochs=50,
       logger=logger
   )

View Results
~~~~~~~~~~~~

.. code-block:: bash

   tensorboard --logdir=./runs

Features
~~~~~~~~

- Automatic metric logging (loss, accuracy, etc.)
- Learning rate tracking
- Model graph visualization
- Image predictions logging
- Automatic run directory creation

MLflow Logger
-------------

Track experiments with MLflow.

Installation
~~~~~~~~~~~~

.. code-block:: bash

   pip install mlflow

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from deepml.tracking import MLFlowLogger

   logger = MLFlowLogger(
       experiment_name='image-classification',
       tracking_uri='./mlruns',  # or remote URI
       log_model_weights=True
   )

   trainer.fit(
       train_loader=train_loader,
       val_loader=val_loader,
       epochs=50,
       logger=logger
   )

View Results
~~~~~~~~~~~~

.. code-block:: bash

   mlflow ui --backend-store-uri ./mlruns

Remote Tracking Server
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   logger = MLFlowLogger(
       experiment_name='my-experiment',
       tracking_uri='http://localhost:5000',
       log_model_weights=True
   )

Features
~~~~~~~~

- Hyperparameter logging
- Metric tracking
- Model artifact storage
- Image logging
- Automatic experiment organization

Weights & Biases Logger
------------------------

Track experiments with Weights & Biases (wandb).

Installation
~~~~~~~~~~~~

.. code-block:: bash

   pip install wandb

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from deepml.tracking import WandbLogger

   logger = WandbLogger(
       project='my-project',
       entity='my-team',  # optional
       name='experiment-1',
       config={
           'learning_rate': 1e-3,
           'batch_size': 32,
           'epochs': 50
       },
       delete_intermediate_artifacts_versions=True
   )

   trainer.fit(
       train_loader=train_loader,
       val_loader=val_loader,
       epochs=50,
       logger=logger
   )

Authentication
~~~~~~~~~~~~~~

.. code-block:: bash

   wandb login

Or set API key:

.. code-block:: python

   import wandb
   wandb.login(key='your-api-key')

Features
~~~~~~~~

- Real-time metric visualization
- Model artifact versioning
- Automatic artifact cleanup (old versions)
- Hyperparameter tracking
- Image and video logging
- Experiment comparison

Custom Logger
-------------

Implement your own logger:

.. code-block:: python

   from deepml.tracking import MLExperimentLogger

   class CustomLogger(MLExperimentLogger):
       def __init__(self, **kwargs):
           super().__init__()
           # Custom initialization

       def log_params(self, **kwargs):
           """Log hyperparameters."""
           print(f"Logging params: {kwargs}")

       def log_metric(self, tag, value, step):
           """Log a scalar metric."""
           print(f"{tag}: {value} at step {step}")

       def log_artifact(self, tag, value, step, artifact_path=None):
           """Log an artifact."""
           pass

       def log_model(self, tag, value, step, artifact_path=None):
           """Log model checkpoint."""
           print(f"Saving model: {artifact_path}")

       def log_image(self, tag, value, step, artifact_path=None):
           """Log an image."""
           pass

   # Use custom logger
   logger = CustomLogger()
   trainer.fit(..., logger=logger)

Comparing Loggers
-----------------

+------------------+---------------+------------------+-------------------+
| Feature          | TensorBoard   | MLflow           | Weights & Biases  |
+==================+===============+==================+===================+
| Setup            | Easy          | Easy             | Requires signup   |
+------------------+---------------+------------------+-------------------+
| Self-hosted      | Yes           | Yes              | Optional          |
+------------------+---------------+------------------+-------------------+
| Real-time        | Yes           | Yes              | Yes               |
+------------------+---------------+------------------+-------------------+
| Model Registry   | No            | Yes              | Yes               |
+------------------+---------------+------------------+-------------------+
| Collaboration    | Limited       | Good             | Excellent         |
+------------------+---------------+------------------+-------------------+
| Artifact Storage | Limited       | Excellent        | Excellent         |
+------------------+---------------+------------------+-------------------+

Logging Images
--------------

Image predictions are automatically logged at the end of each epoch:

.. code-block:: python

   from deepml.transforms import ImageNetInverseTransform

   inverse_transform = ImageNetInverseTransform()

   trainer.fit(
       ...,
       logger=logger,
       image_inverse_transform=inverse_transform,
       logger_img_size=224
   )

Logging Hyperparameters
-----------------------

Hyperparameters are automatically logged from trainer configuration:

.. code-block:: python

   logger = MLFlowLogger(experiment_name='my-exp')

   trainer = FabricTrainer(...)

   # Automatically logs:
   # - Task type
   # - Optimizer
   # - Learning rate
   # - Batch size (inferred from loader)
   # - Number of epochs
   # - Device/precision settings

Manual Logging
~~~~~~~~~~~~~~

.. code-block:: python

   logger.log_params(
       model_name='ResNet50',
       dropout=0.5,
       weight_decay=1e-4,
       custom_param='value'
   )

Logging Custom Metrics
----------------------

.. code-block:: python

   # During training loop
   logger.log_metric('custom_metric', value, step=epoch)

Best Practices
--------------

1. **Consistent Naming**:

   .. code-block:: python

      # Use consistent experiment names
      logger = MLFlowLogger(
          experiment_name='cifar10-resnet50',
          # Add date/version in run name
      )

2. **Tag Experiments**:

   .. code-block:: python

      logger = WandbLogger(
          project='image-classification',
          tags=['baseline', 'resnet', 'v1']
      )

3. **Version Control Integration**:

   .. code-block:: python

      import git

      repo = git.Repo(search_parent_directories=True)
      commit_hash = repo.head.object.hexsha

      logger.log_params(git_commit=commit_hash)

4. **Organize Experiments**:

   .. code-block:: bash

      runs/
      ├── cifar10/
      │   ├── baseline/
      │   ├── augmented/
      │   └── transfer_learning/
      └── imagenet/
          └── resnet50/

5. **Clean Up Old Artifacts**:

   .. code-block:: python

      # WandbLogger automatically cleans up
      logger = WandbLogger(
          delete_intermediate_artifacts_versions=True
      )

Example: Complete Setup
-----------------------

.. code-block:: python

   from deepml.tracking import MLFlowLogger, WandbLogger
   from deepml.fabric_trainer import FabricTrainer
   from deepml.transforms import ImageNetInverseTransform

   # Setup logger
   logger = MLFlowLogger(
       experiment_name='cifar10-classification',
       tracking_uri='./mlruns',
       log_model_weights=True
   )

   # Log custom hyperparameters
   logger.log_params(
       architecture='ResNet50',
       pretrained=True,
       optimizer='Adam',
       weight_decay=1e-4,
       notes='Baseline experiment'
   )

   # Create trainer
   trainer = FabricTrainer(
       task=task,
       optimizer=optimizer,
       criterion=criterion,
       precision='16-mixed'
   )

   # Train with logging
   inverse_transform = ImageNetInverseTransform()

   trainer.fit(
       train_loader=train_loader,
       val_loader=val_loader,
       epochs=50,
       logger=logger,
       image_inverse_transform=inverse_transform,
       logger_img_size=224,
       save_model_after_every_epoch=10
   )

   # Access results
   print(f"Best validation loss: {trainer.best_val_loss}")
   print(f"Training history: {trainer.history}")
