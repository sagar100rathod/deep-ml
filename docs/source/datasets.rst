Datasets
========

deep-ml provides PyTorch Dataset implementations for common data formats.

ImageDataFrameDataset
---------------------

Load images from file paths listed in a pandas DataFrame.

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   import pandas as pd
   from deepml.datasets import ImageDataFrameDataset
   from torchvision import transforms

   # Create DataFrame
   df = pd.DataFrame({
       'image': ['img1.jpg', 'img2.jpg', 'img3.jpg'],
       'label': [0, 1, 2]
   })

   # Create dataset
   transform = transforms.Compose([
       transforms.Resize((224, 224)),
       transforms.ToTensor(),
       transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
   ])

   dataset = ImageDataFrameDataset(
       dataframe=df,
       image_file_name_column='image',
       target_columns='label',
       image_dir='./data/images',
       transforms=transform
   )

Parameters
~~~~~~~~~~

- ``dataframe``: pandas DataFrame with image info
- ``image_file_name_column``: Column containing image filenames (default: ``'image'``)
- ``target_columns``: Column(s) with targets (can be string or list)
- ``image_dir``: Base directory for images (optional)
- ``transforms``: torchvision transforms (optional)
- ``target_transform``: Transforms for targets (optional)
- ``open_file_func``: Custom image loading function (optional)

Multi-Target Example
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   df = pd.DataFrame({
       'image': ['img1.jpg', 'img2.jpg'],
       'class': [0, 1],
       'bbox_x': [10, 20],
       'bbox_y': [30, 40]
   })

   dataset = ImageDataFrameDataset(
       dataframe=df,
       target_columns=['class', 'bbox_x', 'bbox_y'],
       image_dir='./images'
   )

   # Returns: image, tensor([class, bbox_x, bbox_y])

ImageRowDataFrameDataset
-------------------------

For datasets where each row contains a flattened image array (e.g., MNIST CSV).

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from deepml.datasets import ImageRowDataFrameDataset
   import pandas as pd

   # Load data (each row is a flattened image)
   df = pd.read_csv('mnist.csv')

   # First column is label, rest are pixels
   dataset = ImageRowDataFrameDataset(
       dataframe=df,
       target_column='label',
       image_size=(28, 28),
       transform=transforms.ToTensor()
   )

Parameters
~~~~~~~~~~

- ``dataframe``: DataFrame where each row is a flattened image
- ``target_column``: Column containing labels (optional)
- ``image_size``: Tuple (height, width) to reshape images to
- ``transform``: torchvision transforms (optional)

SegmentationDataFrameDataset
-----------------------------

For semantic segmentation with images and mask pairs.

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from deepml.datasets import SegmentationDataFrameDataset
   import albumentations as A
   from albumentations.pytorch import ToTensorV2

   df = pd.DataFrame({
       'image': ['img1.jpg', 'img2.jpg'],
       'mask': ['mask1.png', 'mask2.png']
   })

   # Albumentations transforms (applies to both image and mask)
   transform = A.Compose([
       A.Resize(256, 256),
       A.HorizontalFlip(p=0.5),
       A.Normalize(),
       ToTensorV2()
   ])

   dataset = SegmentationDataFrameDataset(
       dataframe=df,
       image_dir='./images',
       mask_dir='./masks',
       image_col='image',
       mask_col='mask',
       albu_torch_transforms=transform,
       train=True
   )

Parameters
~~~~~~~~~~

- ``dataframe``: DataFrame with image and mask file info
- ``image_dir``: Directory containing images
- ``mask_dir``: Directory containing masks (required for training)
- ``image_col``: Column with image filenames (default: ``'image'``)
- ``mask_col``: Column with mask filenames (default: same as ``image_col``)
- ``albu_torch_transforms``: Albumentations transforms
- ``target_transform``: Additional mask transforms
- ``train``: Training mode (returns masks) or inference mode (returns filenames)
- ``open_file_func``: Custom loading function

Training vs Inference Mode
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Training: returns (image, mask)
   train_dataset = SegmentationDataFrameDataset(
       ...,
       train=True
   )

   # Inference: returns (image, filename)
   test_dataset = SegmentationDataFrameDataset(
       ...,
       mask_dir=None,
       train=False
   )

ImageListDataset
----------------

Load all images from a directory (useful for inference).

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from deepml.datasets import ImageListDataset

   dataset = ImageListDataset(
       image_dir='./unlabeled_images',
       transforms=transform
   )

   # Returns: (image, filename) for each image in directory

Parameters
~~~~~~~~~~

- ``image_dir``: Directory containing images
- ``transforms``: torchvision transforms (optional)
- ``open_file_func``: Custom loading function (optional)

Custom Loading Functions
-------------------------

Override default image loading:

.. code-block:: python

   import cv2
   import numpy as np

   def load_with_opencv(path):
       img = cv2.imread(path)
       img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
       return img

   dataset = ImageDataFrameDataset(
       ...,
       open_file_func=load_with_opencv
   )

For Segmentation (must return numpy arrays):

.. code-block:: python

   def load_seg_image(path):
       from PIL import Image
       return np.array(Image.open(path))

   seg_dataset = SegmentationDataFrameDataset(
       ...,
       open_file_func=load_seg_image
   )

Data Augmentation
-----------------

Using torchvision
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from torchvision import transforms

   train_transform = transforms.Compose([
       transforms.RandomResizedCrop(224),
       transforms.RandomHorizontalFlip(),
       transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
       transforms.ToTensor(),
       transforms.Normalize([0.485, 0.456, 0.406],
                          [0.229, 0.224, 0.225])
   ])

   val_transform = transforms.Compose([
       transforms.Resize(256),
       transforms.CenterCrop(224),
       transforms.ToTensor(),
       transforms.Normalize([0.485, 0.456, 0.406],
                          [0.229, 0.224, 0.225])
   ])

Using Albumentations (for Segmentation)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import albumentations as A
   from albumentations.pytorch import ToTensorV2

   transform = A.Compose([
       A.Resize(512, 512),
       A.HorizontalFlip(p=0.5),
       A.VerticalFlip(p=0.5),
       A.RandomBrightnessContrast(p=0.2),
       A.ShiftScaleRotate(
           shift_limit=0.0625,
           scale_limit=0.1,
           rotate_limit=45,
           p=0.5
       ),
       A.Normalize(
           mean=[0.485, 0.456, 0.406],
           std=[0.229, 0.224, 0.225]
       ),
       ToTensorV2()
   ])

Best Practices
--------------

1. **Use Appropriate Dataset**:

   - File paths in DataFrame → ``ImageDataFrameDataset``
   - Flattened arrays in DataFrame → ``ImageRowDataFrameDataset``
   - Segmentation → ``SegmentationDataFrameDataset``
   - Directory of images → ``ImageListDataset``

2. **Normalization**:

   - Use ImageNet statistics for transfer learning:
     ``mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]``
   - Compute custom statistics for your dataset if training from scratch

3. **Albumentations for Segmentation**:

   - Always use Albumentations for segmentation tasks
   - Ensures image and mask are transformed identically
   - More augmentation options than torchvision

4. **Data Loading Performance**:

   .. code-block:: python

      from torch.utils.data import DataLoader

      loader = DataLoader(
          dataset,
          batch_size=32,
          shuffle=True,
          num_workers=4,      # Parallel loading
          pin_memory=True,    # Faster GPU transfer
          persistent_workers=True  # Keep workers alive
      )

5. **Debugging**:

   .. code-block:: python

      # Check dataset
      image, target = dataset[0]
      print(f"Image shape: {image.shape}")
      print(f"Target: {target}")

      # Visualize augmentations
      import matplotlib.pyplot as plt

      fig, axes = plt.subplots(2, 4, figsize=(16, 8))
      for i in range(8):
          img, _ = dataset[0]
          axes[i//4, i%4].imshow(img.permute(1, 2, 0))
      plt.show()

Example: Complete Pipeline
---------------------------

.. code-block:: python

   import pandas as pd
   from deepml.datasets import ImageDataFrameDataset, SegmentationDataFrameDataset
   from torch.utils.data import DataLoader
   from torchvision import transforms
   import albumentations as A
   from albumentations.pytorch import ToTensorV2

   # Classification
   cls_df = pd.read_csv('classification_data.csv')
   cls_transform = transforms.Compose([
       transforms.Resize(256),
       transforms.CenterCrop(224),
       transforms.ToTensor(),
       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
   ])

   cls_dataset = ImageDataFrameDataset(
       dataframe=cls_df,
       image_dir='./images',
       target_columns='class',
       transforms=cls_transform
   )

   cls_loader = DataLoader(cls_dataset, batch_size=32, shuffle=True)

   # Segmentation
   seg_df = pd.read_csv('segmentation_data.csv')
   seg_transform = A.Compose([
       A.Resize(512, 512),
       A.HorizontalFlip(p=0.5),
       A.Normalize(),
       ToTensorV2()
   ])

   seg_dataset = SegmentationDataFrameDataset(
       dataframe=seg_df,
       image_dir='./images',
       mask_dir='./masks',
       albu_torch_transforms=seg_transform
   )

   seg_loader = DataLoader(seg_dataset, batch_size=4, shuffle=True)
