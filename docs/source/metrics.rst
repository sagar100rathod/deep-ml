Metrics
=======

deep-ml provides built-in metrics for classification and segmentation backed by
`torchmetrics <https://torchmetrics.readthedocs.io>`_. All metrics inherit from
``torch.nn.Module`` and implement a ``forward(output, target)`` method.

.. warning::

   **Breaking changes in v3.3.0** — if you are upgrading:

   - **Classification multiclass**: ``Accuracy()`` → ``Accuracy(num_classes=N)``.
     Without ``num_classes``, binary mode is assumed and a ``ValueError`` is raised
     for ``(N, C)`` shaped predictions.
   - **Classification**: ``epsilon`` parameter removed; use ``zero_division`` instead.
   - **Segmentation default reduction**: changed from ``'macro-imagewise'`` to
     ``'macro'``. Imagewise variants are no longer supported.
   - **Removed classes**: ``BinaryAccuracy``, ``Binarizer``, ``IoU``,
     ``DiceCoefficient``, ``PixelAccuracy``.  Use ``IoUScore``, ``F1Score``,
     ``Accuracy`` instead.


Stateful Epoch-Level Accumulation
----------------------------------

Starting in v3.3.0, all built-in metrics use **stateful epoch-level accumulation**:

- Raw TP, FP, FN counts are accumulated across **every batch** in the epoch.
- The metric value is computed **once** from the global counts at the end of each epoch.
- The progress bar shows the running epoch-to-date value (updated after each batch).

This matters for ratio metrics (IoU, Dice, Precision, Recall, F1) because:

.. math::

   \text{mean}(\text{IoU per batch}) \neq \text{global IoU from all batches}

The old per-batch SMA was Jensen-biased and composition-order dependent. The new
implementation gives the same result as computing the metric on the full dataset.

.. note::

   Built-in metrics are automatically moved to the training device (CUDA, MPS) at
   the start of ``fit()``. No manual ``.to(device)`` call is needed.


Classification Metrics
-----------------------

All classification metrics live in ``deepml.metrics.classification``.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Class
     - Description
   * - ``Accuracy``
     - Fraction of correctly classified samples
   * - ``Precision``
     - TP / (TP + FP)
   * - ``Recall``
     - TP / (TP + FN)
   * - ``FScore``
     - F-beta score (beta=1 → F1, beta=2 → recall-weighted F2)
   * - ``MCC``
     - Matthews Correlation Coefficient — robust for imbalanced datasets

**Constructor parameters (shared)**:

- ``num_classes`` *(int, optional)* — ``None`` → binary; ``int`` → multiclass. **Required for (N, C) shaped predictions.**
- ``threshold`` *(float)* — decision threshold for binary. Default ``0.5``.
- ``zero_division`` *(float)* — value returned when denominator is 0. Default ``0.0``.
- ``beta`` *(float, FScore only)* — beta factor. Default ``1.0``.

.. code-block:: python

   from deepml.metrics.classification import Accuracy, Precision, Recall, FScore, MCC

   # Binary classification — sigmoid output (N,) or (N, 1)
   metrics = {
       "acc": Accuracy(),
       "prec": Precision(),
       "rec": Recall(),
       "f1": FScore(),
       "mcc": MCC(),
   }

   # Multiclass classification (e.g. MNIST with 10 classes)
   metrics = {
       "acc": Accuracy(num_classes=10),
       "prec": Precision(num_classes=10),
       "f2": FScore(num_classes=10, beta=2.0),
       "mcc": MCC(num_classes=10),
   }

   trainer.fit(train_loader, val_loader, epochs=50, metrics=metrics)


Segmentation Metrics
---------------------

All segmentation metrics live in ``deepml.metrics.segmentation``.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Class
     - Description
   * - ``IoUScore``
     - Jaccard index (Intersection over Union)
   * - ``F1Score``
     - Dice coefficient
   * - ``Precision``
     - TP / (TP + FP) per class then reduced
   * - ``Recall``
     - TP / (TP + FN) per class then reduced
   * - ``Accuracy``
     - Pixel accuracy

**Constructor parameters (shared)**:

- ``mode`` *(str)* — ``'binary'``, ``'multiclass'``, or ``'multilabel'``. Default ``'binary'``.
- ``reduction`` *(str)* — ``'macro'`` (default), ``'micro'``, or ``'weighted'``.
- ``num_classes`` *(int)* — required for multiclass/multilabel.
- ``ignore_index`` *(int, optional)* — pixel label to ignore (not supported for binary).
- ``threshold`` *(float, optional)* — binarization threshold for binary/multilabel. Default ``0.5``.
- ``class_weights`` — per-class weights for weighted reduction.
- ``target_class_index`` *(int, optional)* — extract metric for a single class (not supported for binary).
- ``zero_division`` *(float)* — value returned on zero division. Default ``1.0``.
- ``activation`` — custom activation applied to model output before metric computation.
- ``callable`` — preprocessing function ``(output, target) → (output, target)`` applied before activation.

Binary Segmentation
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from deepml.metrics.segmentation import IoUScore, F1Score, Precision, Recall

   metrics = {
       "iou": IoUScore(mode="binary"),
       "dice": F1Score(mode="binary"),
       "precision": Precision(mode="binary"),
       "recall": Recall(mode="binary"),
   }

   trainer.fit(train_loader, val_loader, epochs=50, metrics=metrics)

Multiclass Segmentation
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from deepml.metrics.segmentation import IoUScore, Precision

   metrics = {
       # Global mean IoU across all classes
       "miou": IoUScore(mode="multiclass", num_classes=21, reduction="macro"),

       # Micro IoU (global pixel-level)
       "miou_micro": IoUScore(mode="multiclass", num_classes=21, reduction="micro"),

       # Precision for a single class (e.g. class 0 = background)
       "bg_prec": Precision(
           mode="multiclass", num_classes=21, target_class_index=0
       ),

       # Ignore background class (label 0)
       "fg_iou": IoUScore(mode="multiclass", num_classes=21, ignore_index=0),
   }

Multilabel Segmentation
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from deepml.metrics.segmentation import IoUScore, F1Score

   metrics = {
       "iou": IoUScore(mode="multilabel", num_classes=4, threshold=0.5),
       "dice": F1Score(mode="multilabel", num_classes=4),
   }

Using a Custom Preprocessing Callable
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``callable`` parameter applies before the activation, useful for cropping,
resizing, or masking outputs and targets together:

.. code-block:: python

   def center_crop(output, target):
       # Crop both to remove border artifacts
       return output[..., 8:-8, 8:-8], target[..., 8:-8, 8:-8]

   metrics = {
       "iou": IoUScore(mode="binary", callable=center_crop),
   }


Custom Metrics
--------------

You can pass any ``torch.nn.Module`` with a ``forward(output, target) → scalar``
method as a metric. Custom metrics are treated as non-stateful and aggregated
with a simple moving average across batches.

.. code-block:: python

   import torch
   import torch.nn as nn

   class TopKAccuracy(nn.Module):
       def __init__(self, k=5):
           super().__init__()
           self.k = k

       def forward(self, output, target):
           _, top_k = output.topk(self.k, dim=1)
           correct = top_k.eq(target.unsqueeze(1).expand_as(top_k))
           return correct.any(dim=1).float().mean()

   metrics = {
       "acc": Accuracy(num_classes=1000),
       "top5": TopKAccuracy(k=5),
   }

.. note::

   Custom metrics using manual TP/FP/FN computation are non-stateful and subject
   to the same Jensen bias as the old built-in metrics. Prefer the built-in
   ``IoUScore``, ``Precision``, ``Recall``, ``F1Score`` classes for ratio metrics.


Metric Logging
--------------

Metrics are automatically logged to:

1. **Console** (progress bar) — running epoch-to-date value after each batch
2. **History** (``trainer.history``) — epoch-level value after each epoch
3. **TensorBoard / MLflow** (if logger configured) — under ``{name}/train`` and ``{name}/val``

.. code-block:: python

   trainer.fit(train_loader, val_loader, epochs=50, metrics={"iou": IoUScore(mode="binary")})

   # Access history
   print(trainer.history["train_iou"])   # list of epoch-level IoU values
   print(trainer.history["val_iou"])

   import matplotlib.pyplot as plt
   plt.plot(trainer.history["train_iou"], label="Train IoU")
   plt.plot(trainer.history["val_iou"], label="Val IoU")
   plt.legend()
   plt.show()
