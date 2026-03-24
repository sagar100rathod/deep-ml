Tasks
=====

Tasks define how your model processes data for specific problem types. deep-ml provides
built-in tasks for common computer vision problems.

Task Overview
-------------

All tasks inherit from the abstract ``Task`` base class and implement:

- Data preprocessing (``train_step``, ``eval_step``)
- Output transformation (``transform_output``)
- Prediction methods (``predict``, ``predict_class``)
- Visualization (``show_predictions``)

ImageClassification
-------------------

For single-label image classification problems.

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from deepml.tasks import ImageClassification
   import torch.nn as nn

   task = ImageClassification(
       model=model,
       model_dir='./checkpoints',
       classes=['cat', 'dog', 'bird'],  # Optional class names
       device='auto'
   )

Parameters
~~~~~~~~~~

- ``model``: PyTorch model (``torch.nn.Module``)
- ``model_dir``: Directory for saving checkpoints
- ``load_saved_model``: Resume from checkpoint (default: ``False``)
- ``model_file_name``: Checkpoint filename (default: ``'latest_model.pt'``)
- ``device``: Device selection (``'auto'``, ``'cpu'``, ``'cuda'``, ``'mps'``)
- ``classes``: List of class names for visualization (optional)

Methods
~~~~~~~

**Prediction:**

.. code-block:: python

   # Get raw predictions
   predictions, targets = task.predict(test_loader)

   # Get class labels and probabilities
   predicted_classes, probabilities, targets = task.predict_class(test_loader)

**Visualization:**

.. code-block:: python

   task.show_predictions(
       loader=val_loader,
       image_inverse_transform=denormalize,
       samples=9,
       cols=3,
       figsize=(10, 10),
       target_known=True
   )

Output Format
~~~~~~~~~~~~~

- **Binary Classification**: Sigmoid activation, threshold at 0.5
- **Multiclass Classification**: Softmax activation, argmax for class

MultiLabelImageClassification
------------------------------

For images with multiple labels (multi-hot encoding).

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from deepml.tasks import MultiLabelImageClassification

   task = MultiLabelImageClassification(
       model=model,
       model_dir='./checkpoints',
       classes=['person', 'car', 'tree', 'building']
   )

Example Use Case
~~~~~~~~~~~~~~~~

.. code-block:: python

   # Image can have multiple labels
   # Target: [1, 0, 1, 1] means image contains person, tree, and building

   predicted_classes, probabilities, targets = task.predict_class(test_loader)

   # Probabilities shape: (batch_size, num_classes)
   # predicted_classes: binary (0 or 1) for each class

Output Format
~~~~~~~~~~~~~

- **Activation**: Sigmoid for each class independently
- **Threshold**: 0.5 for each class
- **Output**: Binary vector indicating presence of each class

Segmentation
------------

For semantic segmentation (pixel-level classification).

Binary Segmentation
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from deepml.tasks import Segmentation

   task = Segmentation(
       model=model,
       model_dir='./checkpoints',
       mode='binary',
       num_classes=1,
       threshold=0.5,
       color_map={0: 0, 1: 255}  # Grayscale
   )

Multiclass Segmentation
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   task = Segmentation(
       model=model,
       model_dir='./checkpoints',
       mode='multiclass',
       num_classes=21,  # e.g., Pascal VOC
       color_map={
           0: [0, 0, 0],       # Background
           1: [128, 0, 0],     # Class 1
           2: [0, 128, 0],     # Class 2
           # ... more classes
       }
   )

Parameters
~~~~~~~~~~

- ``mode``: ``'binary'`` or ``'multiclass'``
- ``num_classes``: Number of segmentation classes
- ``threshold``: Probability threshold for binary mode (default: 0.5)
- ``color_map``: Dict mapping class indices to colors

  - Binary: ``{0: 0, 1: 255}`` (grayscale)
  - Multiclass: ``{0: [R, G, B], ...}`` (RGB triplets)

Methods
~~~~~~~

**Save Predictions:**

.. code-block:: python

   task.save_prediction(
       loader=test_loader,
       save_dir='./predictions'
   )
   # Saves PNG files with color-coded masks

**Visualize:**

.. code-block:: python

   task.show_predictions(
       loader=val_loader,
       samples=4,
       cols=2,
       figsize=(16, 16)
   )
   # Shows input, ground truth, and prediction side-by-side

Output Format
~~~~~~~~~~~~~

- **Binary**: Sigmoid + threshold → class indices (0 or 1)
- **Multiclass**: Softmax + argmax → class indices
- **Mask Shape**: (batch, height, width)

ImageRegression
---------------

For predicting continuous values from images (e.g., age, depth, pose).

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from deepml.tasks import ImageRegression

   task = ImageRegression(
       model=model,
       model_dir='./checkpoints'
   )

Example Use Cases
~~~~~~~~~~~~~~~~~

- Age estimation
- Depth prediction
- Pose estimation
- Image quality assessment

Methods
~~~~~~~

.. code-block:: python

   # Get predictions
   predictions, targets = task.predict(test_loader)

   # Predictions are continuous values
   # Shape: (num_samples, output_dim)

   # Visualize
   task.show_predictions(
       loader=val_loader,
       samples=9,
       cols=3
   )

NeuralNetTask
-------------

Generic task for custom implementations.

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from deepml.tasks import NeuralNetTask

   task = NeuralNetTask(
       model=model,
       model_dir='./checkpoints'
   )

Use this when you need:

- Custom prediction logic
- Non-standard output formats
- Specialized preprocessing

Creating Custom Tasks
---------------------

Extend the ``Task`` base class:

.. code-block:: python

   from deepml.tasks import Task
   import torch

   class CustomTask(Task):
       def __init__(self, model, model_dir, **kwargs):
           super().__init__(model, model_dir, **kwargs)
           # Custom initialization

       def transform_target(self, y):
           """Transform target for visualization."""
           return y

       def transform_output(self, prediction):
           """Transform model output."""
           return prediction

       def predict_batch(self, x, *args, **kwargs):
           """Process a single batch."""
           x = self.move_input_to_device(x, **kwargs)
           return self._model(x)

       def train_step(self, x, y, *args, **kwargs):
           """Training step logic."""
           outputs = self.predict_batch(x, *args, **kwargs)
           return outputs, x, y

       def eval_step(self, x, y, *args, **kwargs):
           """Evaluation step logic."""
           outputs = self.predict_batch(x, *args, **kwargs)
           return outputs, x, y

       def predict(self, loader):
           """Generate predictions for entire dataset."""
           # Implementation
           pass

       def predict_class(self, loader):
           """Generate class predictions."""
           # Implementation
           pass

       def show_predictions(self, loader, **kwargs):
           """Visualize predictions."""
           # Implementation
           pass

       def write_prediction_to_logger(self, tag, loader, logger, **kwargs):
           """Log predictions to experiment tracker."""
           # Implementation
           pass

       def evaluate(self, loader, metrics, **kwargs):
           """Evaluate model on metrics."""
           # Implementation
           pass

Task Comparison
---------------

+----------------------------------+-------------------+---------------------+------------------+
| Task                             | Problem Type      | Output Activation   | Typical Loss     |
+==================================+===================+=====================+==================+
| ImageClassification              | Single-label      | Softmax/Sigmoid     | CrossEntropy/BCE |
+----------------------------------+-------------------+---------------------+------------------+
| MultiLabelImageClassification    | Multi-label       | Sigmoid             | BCE              |
+----------------------------------+-------------------+---------------------+------------------+
| Segmentation (binary)            | Pixel binary      | Sigmoid             | BCE/Dice         |
+----------------------------------+-------------------+---------------------+------------------+
| Segmentation (multiclass)        | Pixel multiclass  | Softmax             | CrossEntropy     |
+----------------------------------+-------------------+---------------------+------------------+
| ImageRegression                  | Continuous        | None                | MSE/MAE          |
+----------------------------------+-------------------+---------------------+------------------+

Best Practices
--------------

1. **Choose the Right Task**:

   - Image classification → ``ImageClassification``
   - Multiple labels per image → ``MultiLabelImageClassification``
   - Pixel-level labels → ``Segmentation``
   - Continuous outputs → ``ImageRegression``

2. **Model Output Shape**:

   - Classification: ``(batch, num_classes)``
   - Segmentation: ``(batch, num_classes, height, width)``
   - Regression: ``(batch, output_dim)``

3. **Device Management**:

   - Use ``device='auto'`` for automatic device selection
   - Task automatically moves data to the correct device

4. **Checkpointing**:

   - Always specify ``model_dir`` for saving checkpoints
   - Use ``load_saved_model=True`` to resume training

5. **Visualization**:

   - Provide ``image_inverse_transform`` for proper image display
   - Set appropriate ``samples`` and ``cols`` for your screen
