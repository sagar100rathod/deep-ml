Visualization
=============

deep-ml provides utilities for visualizing data and model predictions.

Visualizing Datasets
--------------------

From DataLoader
~~~~~~~~~~~~~~~

.. code-block:: python

   from deepml.visualize import show_images_from_loader

   show_images_from_loader(
       loader=train_loader,
       image_inverse_transform=denormalize,
       samples=9,
       cols=3,
       figsize=(10, 10),
       classes=['cat', 'dog', 'bird']
   )

From Dataset
~~~~~~~~~~~~

.. code-block:: python

   from deepml.visualize import show_images_from_dataset

   show_images_from_dataset(
       dataset=train_dataset,
       image_inverse_transform=denormalize,
       samples=16,
       cols=4,
       figsize=(12, 12)
   )

From Folder
~~~~~~~~~~~

.. code-block:: python

   from deepml.visualize import show_images_from_folder

   show_images_from_folder(
       img_dir='./data/images',
       samples=9,
       cols=3,
       figsize=(10, 10)
   )

From DataFrame
~~~~~~~~~~~~~~

.. code-block:: python

   from deepml.visualize import show_images_from_dataframe
   import pandas as pd

   df = pd.DataFrame({
       'image': ['img1.jpg', 'img2.jpg', 'img3.jpg'],
       'label': ['cat', 'dog', 'bird']
   })

   show_images_from_dataframe(
       dataframe=df,
       img_dir='./data/images',
       image_file_name_column='image',
       label_column='label',
       samples=9,
       cols=3
   )

Visualizing Predictions
------------------------

Classification
~~~~~~~~~~~~~~

.. code-block:: python

   from deepml.tasks import ImageClassification

   task = ImageClassification(
       model=model,
       model_dir='./checkpoints',
       classes=['cat', 'dog', 'bird']
   )

   # Show predictions
   task.show_predictions(
       loader=val_loader,
       image_inverse_transform=denormalize,
       samples=9,
       cols=3,
       figsize=(12, 12),
       target_known=True
   )

Output:

- Green titles: Correct predictions
- Red titles: Incorrect predictions
- Format: "GT=cat\nPred=dog, 0.85"

Segmentation
~~~~~~~~~~~~

.. code-block:: python

   from deepml.tasks import Segmentation

   task = Segmentation(
       model=model,
       model_dir='./checkpoints',
       mode='binary',
       num_classes=1
   )

   # Show side-by-side comparison
   task.show_predictions(
       loader=val_loader,
       samples=4,
       cols=2,
       figsize=(16, 16)
   )

Output shows:

- Input image
- Ground truth mask
- Predicted mask

Regression
~~~~~~~~~~

.. code-block:: python

   from deepml.tasks import ImageRegression

   task = ImageRegression(
       model=model,
       model_dir='./checkpoints'
   )

   task.show_predictions(
       loader=val_loader,
       samples=9,
       cols=3
   )

Custom Plotting
---------------

Plot Images Grid
~~~~~~~~~~~~~~~~

.. code-block:: python

   from deepml.visualize import plot_images

   images = [img1, img2, img3, img4]
   labels = ['Image 1', 'Image 2', 'Image 3', 'Image 4']

   plot_images(
       images=images,
       labels=labels,
       cols=2,
       figsize=(10, 10),
       fontsize=14
   )

Plot with Colored Titles
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from deepml.visualize import plot_images_with_title

   def image_generator():
       for i in range(9):
           img = get_image(i)
           title = f"Image {i}"
           color = 'green' if i % 2 == 0 else 'red'
           yield img, title, color

   plot_images_with_title(
       image_generator=image_generator(),
       samples=9,
       cols=3,
       figsize=(12, 12)
   )

Plot with Bounding Boxes
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from deepml.visualize import plot_images_with_bboxes

   def bbox_generator():
       for i in range(4):
           img = get_image(i)
           title = f"Image {i}"
           # Bboxes: [class_id, xmin, ymin, width, height]
           bboxes = [
               [0, 10, 20, 50, 60],  # class 0
               [1, 100, 150, 30, 40]  # class 1
           ]
           yield img, title, bboxes

   plot_images_with_bboxes(
       image_generator=bbox_generator(),
       samples=4,
       cols=2,
       classes=['cat', 'dog'],
       class_color_map={0: '#ff0000', 1: '#00ff00'},
       figsize=(12, 12)
   )

Image Transformations
---------------------

Denormalization
~~~~~~~~~~~~~~~

.. code-block:: python

   from deepml.transforms import ImageNetInverseTransform

   # For ImageNet normalization
   denormalize = ImageNetInverseTransform()

   # Custom denormalization
   class CustomDenormalize:
       def __init__(self, mean, std):
           self.mean = torch.tensor(mean).view(-1, 1, 1)
           self.std = torch.tensor(std).view(-1, 1, 1)

       def __call__(self, tensor):
           return tensor * self.std + self.mean

   denormalize = CustomDenormalize(
       mean=[0.485, 0.456, 0.406],
       std=[0.229, 0.224, 0.225]
   )

Tensor to Image
~~~~~~~~~~~~~~~

.. code-block:: python

   import matplotlib.pyplot as plt
   import numpy as np

   def show_tensor(tensor, denormalize=None):
       """Display a tensor as an image."""
       if denormalize:
           tensor = denormalize(tensor)

       # Convert to numpy
       if tensor.dim() == 4:  # Batch
           tensor = tensor[0]

       img = tensor.cpu().numpy()
       img = np.transpose(img, (1, 2, 0))
       img = np.clip(img, 0, 1)

       plt.imshow(img)
       plt.axis('off')
       plt.show()

Saving Visualizations
---------------------

Save Predictions to Files
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import matplotlib.pyplot as plt

   task.show_predictions(
       loader=val_loader,
       samples=9,
       cols=3
   )
   plt.savefig('predictions.png', dpi=300, bbox_inches='tight')
   plt.close()

Save Segmentation Masks
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   task.save_prediction(
       loader=test_loader,
       save_dir='./predictions'
   )
   # Saves color-coded PNG files

Create Prediction Videos
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import cv2
   import numpy as np

   fourcc = cv2.VideoWriter_fourcc(*'mp4v')
   out = cv2.VideoWriter('predictions.mp4', fourcc, 10, (640, 480))

   for images, _ in test_loader:
       predictions = model(images)

       for i in range(len(images)):
           img = images[i].cpu().numpy()
           pred = predictions[i].cpu().numpy()

           # Create visualization
           vis = create_visualization(img, pred)
           out.write(vis)

   out.release()

Advanced Visualization
----------------------

Attention Maps
~~~~~~~~~~~~~~

.. code-block:: python

   import torch.nn.functional as F

   def visualize_attention(image, attention_weights):
       """Overlay attention map on image."""
       # Resize attention to image size
       attention = F.interpolate(
           attention_weights.unsqueeze(0).unsqueeze(0),
           size=image.shape[-2:],
           mode='bilinear'
       )

       # Normalize
       attention = (attention - attention.min()) / (attention.max() - attention.min())

       # Create heatmap
       import matplotlib.cm as cm
       heatmap = cm.jet(attention[0, 0].cpu().numpy())[:, :, :3]

       # Overlay
       image_np = image.cpu().permute(1, 2, 0).numpy()
       overlay = 0.6 * image_np + 0.4 * heatmap

       plt.imshow(overlay)
       plt.axis('off')
       plt.show()

Feature Maps
~~~~~~~~~~~~

.. code-block:: python

   def visualize_feature_maps(features, num_maps=16):
       """Visualize intermediate feature maps."""
       fig, axes = plt.subplots(4, 4, figsize=(12, 12))

       for i, ax in enumerate(axes.flat):
           if i < num_maps:
               feature = features[0, i].cpu().detach().numpy()
               ax.imshow(feature, cmap='viridis')
               ax.set_title(f'Map {i}')
           ax.axis('off')

       plt.tight_layout()
       plt.show()

Grad-CAM
~~~~~~~~

.. code-block:: python

   from pytorch_grad_cam import GradCAM
   from pytorch_grad_cam.utils.image import show_cam_on_image

   # Setup Grad-CAM
   target_layer = model.layer4[-1]
   cam = GradCAM(model=model, target_layers=[target_layer])

   # Generate CAM
   grayscale_cam = cam(input_tensor=image)

   # Visualize
   visualization = show_cam_on_image(
       rgb_img,
       grayscale_cam,
       use_rgb=True
   )

   plt.imshow(visualization)
   plt.show()

Best Practices
--------------

1. **Always Denormalize**:

   .. code-block:: python

      # Images won't display correctly if normalized
      show_images_from_loader(
          loader=val_loader,
          image_inverse_transform=denormalize  # Important!
      )

2. **Use Appropriate Grid Size**:

   .. code-block:: python

      # For detailed view
      samples=4, cols=2

      # For overview
      samples=16, cols=4

3. **Save High-Quality Images**:

   .. code-block:: python

      plt.savefig('plot.png', dpi=300, bbox_inches='tight')

4. **Close Plots in Loops**:

   .. code-block:: python

      for epoch in range(epochs):
          # ... training ...
          task.show_predictions(...)
          plt.savefig(f'epoch_{epoch}.png')
          plt.close()  # Prevent memory leak

5. **Use Consistent Color Schemes**:

   .. code-block:: python

      # Segmentation color map
      color_map = {
          0: [0, 0, 0],      # Background: black
          1: [255, 0, 0],    # Class 1: red
          2: [0, 255, 0]     # Class 2: green
      }
