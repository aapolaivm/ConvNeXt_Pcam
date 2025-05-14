# ConvNeXt for Medical Image Classification

This repository contains the implementation of a ConvNeXt-based deep learning model for medical image classification, created as part of a Master's thesis project. The application focuses on using the state-of-the-art ConvNeXt architecture to classify medical images, specifically using the PatchCamelyon (PCAM) dataset for cancer detection.

## Project Overview

This project implements a deep learning solution for automated detection of metastases in histological patches using the ConvNeXt architecture. The model is trained to identify cancer metastases in lymph node sections from the PatchCamelyon dataset.

## Requirements

The application automatically installs the following dependencies if they are not already present:
- PyTorch
- TorchVision
- Timm (PyTorch Image Models)
- Scikit-learn
- Pandas
- NumPy
- Gdown
- tqdm

## Installation

1. Clone this repository:
```bash
git clone https://github.com/aapolaivm/ConvNeXt_sovellus.git
cd ConvNeXt_sovellus
```

2. Run the application:
```bash
python convnext.py
```

The script will automatically check for and install required packages if they are missing.

## Usage

### Training a Model

To train the ConvNeXt model on the PCAM dataset, simply run:
```bash
python convnext.py
```

The script will:
1. Download the PCAM dataset if it's not already present in the `data` directory
2. Initialize a ConvNeXt Large model pretrained on ImageNet
3. Train the model for the specified number of epochs
4. Validate performance using AUC-ROC metric
5. Save the best model based on validation performance to the `models` directory

### Using a Different Dataset

To use a different dataset, you need to modify the data loading part of the script. You can either:

1. Use another torchvision dataset by replacing the PCAM dataset with another built-in dataset
2. Create a custom dataset class that inherits from `torch.utils.data.Dataset`

Example for using CIFAR-10:
```python
from torchvision.datasets import CIFAR10

# Transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Data loading
train_dataset = CIFAR10(root="./data", train=True, transform=transform, download=True)
val_dataset = CIFAR10(root="./data", train=False, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Update model class count
model = timm.create_model("convnext_large", pretrained=True, num_classes=10)
```

### Visualizing Results

To visualize the results of your trained model, you can use the `visualize_results.py` script:
```bash
python visualize_results.py
```

## Project Structure

- `convnext.py`: Main script for model training and evaluation
- `custom_dataset.py`: Helper script for creating custom datasets
- `visualize_results.py`: Script for visualizing model results
- `models/`: Directory for saved model weights
- `data/`: Directory for datasets
- `results/`: Directory for output results and visualizations

## License

This project is part of a Master's thesis and is provided for educational purposes.

## Acknowledgments

- The project uses the ConvNeXt architecture from the paper "A ConvNet for the 2020s" (Liu et al., 2022)
- The PatchCamelyon dataset is used for training and evaluation