Tutorials
=========

Step-by-step guides for common use cases.

Tutorial 1: Image Classification
---------------------------------

Train a ResNet model on CIFAR-10 from scratch.

Full Code
~~~~~~~~~

.. code-block:: python

   import torch
   from torch.utils.data import DataLoader
   from torchvision import datasets, transforms
   from torchvision.models import resnet18
   from torch.optim import Adam
   from torch.optim.lr_scheduler import CosineAnnealingLR

   from deepml.tasks import ImageClassification
   from deepml.fabric_trainer import FabricTrainer
   from deepml.metrics.classification import Accuracy
   from deepml.tracking import TensorboardLogger

   # 1. Data preparation
   transform_train = transforms.Compose([
       transforms.RandomCrop(32, padding=4),
       transforms.RandomHorizontalFlip(),
       transforms.ToTensor(),
       transforms.Normalize((0.4914, 0.4822, 0.4465),
                          (0.2023, 0.1994, 0.2010))
   ])

   transform_val = transforms.Compose([
       transforms.ToTensor(),
       transforms.Normalize((0.4914, 0.4822, 0.4465),
                          (0.2023, 0.1994, 0.2010))
   ])

   train_dataset = datasets.CIFAR10(
       root='./data',
       train=True,
       download=True,
       transform=transform_train
   )

   val_dataset = datasets.CIFAR10(
       root='./data',
       train=False,
       download=True,
       transform=transform_val
   )

   train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
   val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4)

   # 2. Model and task
   model = resnet18(num_classes=10)

   classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
              'dog', 'frog', 'horse', 'ship', 'truck']

   task = ImageClassification(
       model=model,
       model_dir='./checkpoints/cifar10',
       classes=classes
   )

   # 3. Training configuration
   optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
   criterion = torch.nn.CrossEntropyLoss()

   lr_scheduler_fn = lambda opt: CosineAnnealingLR(opt, T_max=100)

   # 4. Metrics and logger
   metrics = {'accuracy': Accuracy()}
   logger = TensorboardLogger(model_dir='./runs/cifar10')

   # 5. Create trainer
   trainer = FabricTrainer(
       task=task,
       optimizer=optimizer,
       criterion=criterion,
       lr_scheduler_fn=lr_scheduler_fn,
       accelerator='auto',
       precision='16-mixed'
   )

   # 6. Train
   trainer.fit(
       train_loader=train_loader,
       val_loader=val_loader,
       epochs=100,
       metrics=metrics,
       logger=logger,
       save_model_after_every_epoch=20
   )

   # 7. Visualize results
   task.show_predictions(
       loader=val_loader,
       samples=9,
       cols=3
   )

   print(f"Best validation loss: {trainer.best_val_loss:.4f}")

Tutorial 2: Transfer Learning
------------------------------

Fine-tune a pre-trained model on a custom dataset.

.. code-block:: python

   import torch
   from torchvision.models import resnet50, ResNet50_Weights

   # 1. Load pre-trained model
   model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

   # 2. Freeze backbone
   for param in model.parameters():
       param.requires_grad = False

   # 3. Replace classifier
   num_classes = 5  # Your number of classes
   model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

   # 4. Use ImageNet transforms
   from torchvision.transforms import v2

   transform = v2.Compose([
       v2.Resize(256),
       v2.CenterCrop(224),
       v2.ToTensor(),
       v2.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
   ])

   # 5. Train only the classifier first
   optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)

   trainer.fit(..., epochs=10)

   # 6. Unfreeze and fine-tune entire network
   for param in model.parameters():
       param.requires_grad = True

   optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

   trainer.fit(..., epochs=20)

Tutorial 3: Semantic Segmentation
----------------------------------

Train U-Net for binary segmentation.

.. code-block:: python

   import torch
   import pandas as pd
   import albumentations as A
   from albumentations.pytorch import ToTensorV2

   from deepml.tasks import Segmentation
   from deepml.datasets import SegmentationDataFrameDataset
   from deepml.fabric_trainer import FabricTrainer
   from deepml.losses import JaccardLoss
   from deepml.metrics.segmentation import IoU, DiceCoefficient

   # 1. Prepare data
   df = pd.read_csv('segmentation_data.csv')

   transform = A.Compose([
       A.Resize(512, 512),
       A.HorizontalFlip(p=0.5),
       A.Normalize(),
       ToTensorV2()
   ])

   train_dataset = SegmentationDataFrameDataset(
       dataframe=df[df['split'] == 'train'],
       image_dir='./images',
       mask_dir='./masks',
       albu_torch_transforms=transform
   )

   val_dataset = SegmentationDataFrameDataset(
       dataframe=df[df['split'] == 'val'],
       image_dir='./images',
       mask_dir='./masks',
       albu_torch_transforms=transform
   )

   # 2. Define model (example UNet)
   from segmentation_models_pytorch import Unet

   model = Unet(
       encoder_name='resnet34',
       encoder_weights='imagenet',
       in_channels=3,
       classes=1
   )

   # 3. Create task
   task = Segmentation(
       model=model,
       model_dir='./checkpoints/segmentation',
       mode='binary',
       num_classes=1,
       threshold=0.5
   )

   # 4. Setup training
   optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
   criterion = torch.nn.BCEWithLogitsLoss()

   metrics = {
       'iou': IoU(num_classes=1, is_multiclass=False),
       'dice': DiceCoefficient(num_classes=1, is_multiclass=False)
   }

   # 5. Train
   trainer = FabricTrainer(
       task=task,
       optimizer=optimizer,
       criterion=criterion
   )

   trainer.fit(
       train_loader=DataLoader(train_dataset, batch_size=8, shuffle=True),
       val_loader=DataLoader(val_dataset, batch_size=8),
       epochs=100,
       metrics=metrics
   )

Tutorial 4: Multi-GPU Training
-------------------------------

Distributed training across multiple GPUs.

.. code-block:: python

   from deepml.fabric_trainer import FabricTrainer

   # Option 1: DataParallel (single node)
   trainer = FabricTrainer(
       task=task,
       optimizer=optimizer,
       criterion=criterion,
       accelerator='gpu',
       strategy='dp',
       devices=2  # Use 2 GPUs
   )

   # Option 2: DistributedDataParallel (recommended)
   trainer = FabricTrainer(
       task=task,
       optimizer=optimizer,
       criterion=criterion,
       accelerator='gpu',
       strategy='ddp',
       devices='auto'  # Use all available GPUs
   )

   # Option 3: FSDP for large models
   trainer = FabricTrainer(
       task=task,
       optimizer=optimizer,
       criterion=criterion,
       accelerator='gpu',
       strategy='fsdp',
       devices='auto'
   )

   # Train normally
   trainer.fit(...)

Tutorial 5: Hyperparameter Tuning
----------------------------------

Use Optuna for hyperparameter optimization.

.. code-block:: python

   import optuna

   def objective(trial):
       # Sample hyperparameters
       lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
       batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
       weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)

       # Create dataloaders
       train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
       val_loader = DataLoader(val_dataset, batch_size=batch_size)

       # Create model and optimizer
       model = resnet18(num_classes=10)
       optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

       # Create task and trainer
       task = ImageClassification(model=model, model_dir='./temp')
       trainer = FabricTrainer(task=task, optimizer=optimizer, criterion=criterion)

       # Train for a few epochs
       trainer.fit(train_loader, val_loader, epochs=10)

       # Return validation loss
       return trainer.best_val_loss

   # Run optimization
   study = optuna.create_study(direction='minimize')
   study.optimize(objective, n_trials=20)

   print(f"Best params: {study.best_params}")
   print(f"Best value: {study.best_value}")

Tutorial 6: Model Deployment
-----------------------------

Export and deploy your trained model.

Export to TorchScript
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Load trained model
   task = ImageClassification(
       model=model,
       model_dir='./checkpoints',
       load_saved_model=True
   )

   # Export to TorchScript
   model.eval()
   example_input = torch.randn(1, 3, 224, 224)
   traced_model = torch.jit.trace(model, example_input)
   traced_model.save('model.pt')

   # Load and use
   loaded_model = torch.jit.load('model.pt')
   output = loaded_model(input_tensor)

Export to ONNX
~~~~~~~~~~~~~~

.. code-block:: python

   import torch.onnx

   model.eval()
   dummy_input = torch.randn(1, 3, 224, 224)

   torch.onnx.export(
       model,
       dummy_input,
       'model.onnx',
       input_names=['input'],
       output_names=['output'],
       dynamic_axes={
           'input': {0: 'batch_size'},
           'output': {0: 'batch_size'}
       }
   )

Simple Inference Server
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from flask import Flask, request, jsonify
   from PIL import Image
   import io

   app = Flask(__name__)

   # Load model
   model = torch.jit.load('model.pt')
   model.eval()

   @app.route('/predict', methods=['POST'])
   def predict():
       # Get image
       file = request.files['image']
       img = Image.open(io.BytesIO(file.read()))

       # Preprocess
       img_tensor = transform(img).unsqueeze(0)

       # Predict
       with torch.no_grad():
           output = model(img_tensor)
           pred = output.argmax(dim=1).item()

       return jsonify({'prediction': pred})

   app.run(host='0.0.0.0', port=5000)

Next Steps
----------

- Explore the :doc:`trainers` documentation for advanced training options
- Learn about :doc:`tasks` for different problem types
- Check the :doc:`examples` for more complete projects
- Read the :doc:`faq` for common questions
