Metrics
=======

deep-ml provides metrics for evaluating model performance during training.

All metrics inherit from ``torch.nn.Module`` and implement a ``forward()`` method.

Classification Metrics
----------------------

Accuracy
~~~~~~~~

Simple accuracy metric for classification.

.. code-block:: python

   from deepml.metrics.classification import Accuracy

   metrics = {
       'accuracy': Accuracy()
   }

   trainer.fit(
       train_loader=train_loader,
       val_loader=val_loader,
       epochs=50,
       metrics=metrics
   )

Binary Accuracy
~~~~~~~~~~~~~~~

For binary classification with specific threshold.

.. code-block:: python

   from deepml.metrics.classification import BinaryAccuracy

   metrics = {
       'accuracy': BinaryAccuracy(threshold=0.5)
   }

Segmentation Metrics
--------------------

IoU (Intersection over Union)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Jaccard Index for segmentation evaluation.

.. code-block:: python

   from deepml.metrics.segmentation import IoU

   # Binary segmentation
   metrics = {
       'iou': IoU(num_classes=1, is_multiclass=False)
   }

   # Multiclass segmentation
   metrics = {
       'iou': IoU(num_classes=21, is_multiclass=True)
   }

Dice Coefficient
~~~~~~~~~~~~~~~~

F1-score for segmentation.

.. code-block:: python

   from deepml.metrics.segmentation import DiceCoefficient

   metrics = {
       'dice': DiceCoefficient(num_classes=1, is_multiclass=False)
   }

Pixel Accuracy
~~~~~~~~~~~~~~

Percentage of correctly classified pixels.

.. code-block:: python

   from deepml.metrics.segmentation import PixelAccuracy

   metrics = {
       'pixel_acc': PixelAccuracy()
   }

Custom Metrics
--------------

Create your own metrics by subclassing ``torch.nn.Module``:

.. code-block:: python

   import torch
   import torch.nn as nn

   class F1Score(nn.Module):
       def __init__(self, num_classes):
           super().__init__()
           self.num_classes = num_classes

       def forward(self, predictions, targets):
           """
           Args:
               predictions: Model predictions (logits or probabilities)
               targets: Ground truth labels

           Returns:
               F1 score as a scalar tensor
           """
           # Convert predictions to class indices
           preds = torch.argmax(predictions, dim=1)

           # Calculate F1 per class and average
           f1_scores = []
           for c in range(self.num_classes):
               pred_c = (preds == c)
               target_c = (targets == c)

               tp = (pred_c & target_c).sum().float()
               fp = (pred_c & ~target_c).sum().float()
               fn = (~pred_c & target_c).sum().float()

               precision = tp / (tp + fp + 1e-7)
               recall = tp / (tp + fn + 1e-7)
               f1 = 2 * precision * recall / (precision + recall + 1e-7)
               f1_scores.append(f1)

           return torch.stack(f1_scores).mean()

   # Use in training
   metrics = {
       'f1': F1Score(num_classes=10)
   }

Using Multiple Metrics
----------------------

.. code-block:: python

   from deepml.metrics.classification import Accuracy
   from deepml.metrics.segmentation import IoU, DiceCoefficient

   # Classification
   cls_metrics = {
       'accuracy': Accuracy(),
       'top5_acc': TopKAccuracy(k=5)
   }

   # Segmentation
   seg_metrics = {
       'iou': IoU(num_classes=21, is_multiclass=True),
       'dice': DiceCoefficient(num_classes=21, is_multiclass=True),
       'pixel_acc': PixelAccuracy()
   }

   trainer.fit(
       ...,
       metrics=seg_metrics
   )

Metric Logging
--------------

Metrics are automatically logged to:

1. **Console** (progress bar)
2. **History** (``trainer.history``)
3. **TensorBoard** (if logger is configured)

.. code-block:: python

   trainer.fit(...)

   # Access training history
   print(trainer.history['train_loss'])
   print(trainer.history['train_accuracy'])
   print(trainer.history['val_loss'])
   print(trainer.history['val_accuracy'])

   # Plot metrics
   import matplotlib.pyplot as plt

   plt.plot(trainer.history['train_loss'], label='Train Loss')
   plt.plot(trainer.history['val_loss'], label='Val Loss')
   plt.legend()
   plt.show()

Best Practices
--------------

1. **Keep Metrics Simple**:

   - Metrics are computed every batch
   - Complex metrics slow down training
   - Use simple metrics during training, detailed metrics for evaluation

2. **Return Scalars**:

   .. code-block:: python

      def forward(self, predictions, targets):
          # Always return a scalar
          return metric_value.mean()

3. **Handle Edge Cases**:

   .. code-block:: python

      def forward(self, predictions, targets):
          # Avoid division by zero
          denominator = tp + fp + 1e-7
          precision = tp / denominator
          return precision

4. **Use GPU-Compatible Operations**:

   .. code-block:: python

      # Good: tensor operations
      accuracy = (predictions == targets).float().mean()

      # Avoid: numpy or Python loops
      # accuracy = np.mean(predictions.cpu().numpy() == targets.cpu().numpy())

5. **Metric Selection**:

   - **Classification**: Accuracy, F1, Precision, Recall
   - **Segmentation**: IoU, Dice, Pixel Accuracy
   - **Regression**: MSE, MAE, R²
   - **Imbalanced Data**: F1, Precision/Recall, mAP

Example: Complete Metrics Setup
--------------------------------

.. code-block:: python

   import torch
   import torch.nn as nn
   from deepml.metrics.classification import Accuracy
   from deepml.metrics.segmentation import IoU, DiceCoefficient

   # Custom metric
   class MeanIoU(nn.Module):
       def __init__(self, num_classes):
           super().__init__()
           self.num_classes = num_classes

       def forward(self, predictions, targets):
           preds = torch.argmax(predictions, dim=1)
           ious = []

           for c in range(self.num_classes):
               pred_c = (preds == c)
               target_c = (targets == c)

               intersection = (pred_c & target_c).sum().float()
               union = (pred_c | target_c).sum().float()

               iou = intersection / (union + 1e-7)
               ious.append(iou)

           return torch.stack(ious).mean()

   # Combine built-in and custom metrics
   metrics = {
       'accuracy': Accuracy(),
       'iou': IoU(num_classes=21, is_multiclass=True),
       'dice': DiceCoefficient(num_classes=21, is_multiclass=True),
       'mean_iou': MeanIoU(num_classes=21)
   }

   trainer.fit(
       train_loader=train_loader,
       val_loader=val_loader,
       epochs=100,
       metrics=metrics
   )
