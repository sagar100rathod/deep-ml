.. deepml documentation master file, created by
   sphinx-quickstart on Sat Oct 30 19:58:55 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to deepml's documentation!
==================================

**deepml** is a Python library for quickly training deep learning
models for Computer Vision tasks such as **Image Classification**, **Image Regression**
and **Semantic Segmentation** (Binary and Multiclass) using *PyTorch* framework.

It helps you avoid lots of boilerplate code everytime you write in *PyTorch*
so that you can focus on *training*, *validating* and *visualizing* model's prediction.

The best part is, it offers a *simple* and *intuitive* API.

Silent Features:
=====================

    1. Easy to use wrapper around PyTorch framework so that you can focus on training and validating your model.

    2. Integrates with Tensorboard to use it to monitor metrics while model trains.

    3. Quickly visualize your model's predictions.

    4. Following are different types of machine learning tasks available to choose from **deepml.tasks** module:

        i. ImageClassification.
        ii. MultiLabelImageClassification.
        iii. ImageRegression.
        iv. Segmentation.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   usage
   api


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
