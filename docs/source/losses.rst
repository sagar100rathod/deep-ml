Losses
======

deep-ml provides custom loss functions for computer vision tasks.

Jaccard Loss (IoU Loss)
------------------------

Intersection over Union loss for segmentation tasks.

.. code-block:: python

   from deepml.losses import JaccardLoss

   # Binary segmentation
   criterion = JaccardLoss(is_multiclass=False)

   # Multiclass segmentation
   criterion = JaccardLoss(is_multiclass=True)

Mathematical Definition
~~~~~~~~~~~~~~~~~~~~~~~

.. math::

   \text{Jaccard Loss} = 1 - \frac{\text{Intersection}}{\text{Union}}

   \text{IoU} = \frac{|A \cap B|}{|A \cup B|}

where :math:`A` is the predicted mask and :math:`B` is the ground truth.

Usage Example
~~~~~~~~~~~~~

.. code-block:: python

   import torch
   from deepml.losses import JaccardLoss

   criterion = JaccardLoss(is_multiclass=False)

   # Predictions shape: (batch, channels, height, width)
   predictions = model(images)

   # Targets shape: (batch, channels, height, width)
   loss = criterion(predictions, targets)

RMSE Loss
---------

Root Mean Squared Error for regression tasks.

.. code-block:: python

   from deepml.losses import RMSELoss

   criterion = RMSELoss(eps=1e-6)

Mathematical Definition
~~~~~~~~~~~~~~~~~~~~~~~

.. math::

   \text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2 + \epsilon}

where :math:`\epsilon` is added for numerical stability.

Weighted BCE With Logits Loss
------------------------------

Binary Cross-Entropy with separate weights for positive and negative samples.

.. code-block:: python

   from deepml.losses import WeightedBCEWithLogitsLoss

   # Handle class imbalance
   criterion = WeightedBCEWithLogitsLoss(
       w_p=2.0,  # Weight for positive class
       w_n=1.0   # Weight for negative class
   )

Use Case
~~~~~~~~

Useful for imbalanced binary segmentation:

.. code-block:: python

   # If positive pixels are rare (e.g., 5% of image)
   # Give them higher weight
   criterion = WeightedBCEWithLogitsLoss(w_p=10.0, w_n=1.0)

   loss = criterion(logits, targets)

Contrastive Loss
----------------

For siamese networks and metric learning.

.. code-block:: python

   from deepml.losses import ContrastiveLoss

   criterion = ContrastiveLoss(
       margin=2.0,
       distance_func=None,  # Uses pairwise Euclidean distance
       label_transform=None
   )

Usage Example
~~~~~~~~~~~~~

.. code-block:: python

   # Embeddings from siamese network
   embeddings1 = model(image1)
   embeddings2 = model(image2)

   # Labels: 1 for similar pairs, 0 for dissimilar
   labels = torch.tensor([1, 0, 1, 0])

   loss = criterion((embeddings1, embeddings2), labels)

Mathematical Definition
~~~~~~~~~~~~~~~~~~~~~~~

.. math::

   L = y \cdot d^2 + (1-y) \cdot \max(0, m - d)^2

where:
- :math:`d` is the distance between embeddings
- :math:`y` is 1 for similar pairs, 0 for dissimilar
- :math:`m` is the margin

Angular Penalty Softmax Loss
-----------------------------

For face recognition and metric learning. Implements ArcFace, SphereFace, and CosFace.

.. code-block:: python

   from deepml.losses import AngularPenaltySMLoss

   # ArcFace (recommended)
   criterion = AngularPenaltySMLoss(
       in_features=512,
       out_features=10,
       loss_type='arcface',
       s=64.0,
       m=0.5
   )

   # SphereFace
   criterion = AngularPenaltySMLoss(
       in_features=512,
       out_features=10,
       loss_type='sphereface',
       s=64.0,
       m=1.35
   )

   # CosFace
   criterion = AngularPenaltySMLoss(
       in_features=512,
       out_features=10,
       loss_type='cosface',
       s=30.0,
       m=0.4
   )

Usage Example
~~~~~~~~~~~~~

.. code-block:: python

   import torch.nn as nn

   class FaceRecognitionModel(nn.Module):
       def __init__(self):
           super().__init__()
           self.backbone = ResNet()
           self.fc = nn.Linear(2048, 512)  # Feature embeddings

       def forward(self, x):
           features = self.backbone(x)
           embeddings = self.fc(features)
           return embeddings

   model = FaceRecognitionModel()
   criterion = AngularPenaltySMLoss(
       in_features=512,
       out_features=num_classes,
       loss_type='arcface'
   )

   # Training
   embeddings = model(images)
   loss = criterion(embeddings, labels)

Loss Type Comparison
~~~~~~~~~~~~~~~~~~~~

+-------------+------------------+------------+------------+
| Loss Type   | Margin Type      | Default s  | Default m  |
+=============+==================+============+============+
| ArcFace     | Additive Angular | 64.0       | 0.5        |
+-------------+------------------+------------+------------+
| SphereFace  | Multiplicative   | 64.0       | 1.35       |
+-------------+------------------+------------+------------+
| CosFace     | Additive Cosine  | 30.0       | 0.4        |
+-------------+------------------+------------+------------+

Combining Losses
----------------

Weighted Combination
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from deepml.losses import JaccardLoss
   import torch.nn as nn

   class CombinedLoss(nn.Module):
       def __init__(self):
           super().__init__()
           self.bce = nn.BCEWithLogitsLoss()
           self.iou = JaccardLoss(is_multiclass=False)

       def forward(self, pred, target):
           bce_loss = self.bce(pred, target)
           iou_loss = self.iou(pred, target)
           return 0.5 * bce_loss + 0.5 * iou_loss

   criterion = CombinedLoss()

Multi-Task Loss
~~~~~~~~~~~~~~~

.. code-block:: python

   class MultiTaskLoss(nn.Module):
       def __init__(self):
           super().__init__()
           self.seg_loss = JaccardLoss(is_multiclass=False)
           self.cls_loss = nn.CrossEntropyLoss()

       def forward(self, seg_pred, cls_pred, seg_target, cls_target):
           seg_loss = self.seg_loss(seg_pred, seg_target)
           cls_loss = self.cls_loss(cls_pred, cls_target)
           return seg_loss + 0.5 * cls_loss

Loss Selection Guide
--------------------

Classification
~~~~~~~~~~~~~~

- **Binary**: ``nn.BCEWithLogitsLoss`` or ``WeightedBCEWithLogitsLoss``
- **Multiclass**: ``nn.CrossEntropyLoss``
- **Multi-label**: ``nn.BCEWithLogitsLoss``
- **Face Recognition**: ``AngularPenaltySMLoss``

Segmentation
~~~~~~~~~~~~

- **Binary**: ``nn.BCEWithLogitsLoss`` + ``JaccardLoss``
- **Multiclass**: ``nn.CrossEntropyLoss`` + ``JaccardLoss``
- **Imbalanced**: ``WeightedBCEWithLogitsLoss`` or ``DiceLoss``

Regression
~~~~~~~~~~

- **General**: ``nn.MSELoss``
- **Robust**: ``nn.L1Loss`` (MAE)
- **Root MSE**: ``RMSELoss``

Metric Learning
~~~~~~~~~~~~~~~

- **Siamese Networks**: ``ContrastiveLoss``
- **Face Recognition**: ``AngularPenaltySMLoss`` (ArcFace recommended)

Best Practices
--------------

1. **Use LogitsLoss Variants**:

   - Prefer ``BCEWithLogitsLoss`` over ``BCELoss``
   - Numerically more stable
   - Don't apply sigmoid before loss

2. **Handle Class Imbalance**:

   .. code-block:: python

      # Option 1: Weighted loss
      pos_weight = torch.tensor([num_neg / num_pos])
      criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

      # Option 2: Custom weights
      criterion = WeightedBCEWithLogitsLoss(
          w_p=num_neg / num_pos,
          w_n=1.0
      )

3. **Combine Losses for Segmentation**:

   .. code-block:: python

      # IoU alone may be noisy
      # Combine with CE or BCE
      total_loss = ce_loss + iou_loss

4. **Label Smoothing**:

   .. code-block:: python

      criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

5. **Reduce Mode**:

   .. code-block:: python

      # Default: 'mean'
      criterion = nn.CrossEntropyLoss(reduction='mean')

      # For custom weighting
      criterion = nn.CrossEntropyLoss(reduction='none')
      weighted_loss = (loss * weights).mean()
