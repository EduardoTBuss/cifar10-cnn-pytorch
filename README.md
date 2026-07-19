# CIFAR-10 CNN Classifier

> A deep convolutional neural network for CIFAR-10 image classification, implemented in PyTorch with a modern architecture and strong regularization techniques.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

## Overview

This project implements a deep CNN that classifies the 10 CIFAR-10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.

## Network Architecture

- **5 convolutional blocks** with increasing filters (32 → 64 → 128 → 256 → 512)
- **Batch Normalization** to stabilize training
- **Dropout** for regularization (0.25, 0.3, 0.5)
- **MaxPooling** for spatial downsampling
- **Fully connected classifier** with 1024 neurons

### Regularization Techniques

- Data augmentation (random crops, horizontal flips, AutoAugment)
- Normalization with CIFAR-10 dataset statistics
- Learning rate scheduling (exponential decay)
- Early stopping based on validation accuracy

## Project Structure

```
CNNforCIFAR10/
├── config.py          # Configuration and hyperparameters
├── data.py            # Data loading and preprocessing
├── model.py           # CNN architecture definition
├── train.py           # Training and evaluation loop
├── plot.py            # Result visualization
├── main.py            # Main script
└── README.md          # This file
```

## Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| Learning rate | 0.001 | Initial learning rate |
| Batch size | 16 | Mini-batch size |
| Gamma | 0.97 | Learning rate decay factor |
| Epochs | 100 | Number of epochs |
| Image size | 32×32 | Input dimensions |

## Usage

### Prerequisites

```bash
pip install torch torchvision matplotlib
```

### Training

```bash
python main.py
```

The script will:
1. Automatically detect GPU availability
2. Download the CIFAR-10 dataset (first run)
3. Train the model with live progress reporting
4. Save the best model as `best_model.pth`
5. Generate training plots in `training_plot.png`

### Sample Output

```
Using device: cuda
Epoch [1/100] | Train Loss: 1.8234 | Val Loss: 1.6543 | Val Acc: 42.15%
Epoch [2/100] | Train Loss: 1.5678 | Val Loss: 1.4321 | Val Acc: 48.72%
...
Best validation accuracy: 89.45%
```

## Technical Details

- **Data augmentation**: RandomCrop with padding, RandomHorizontalFlip, AutoAugment (CIFAR-10 policy)
- **Optimizer**: AdamW
- **Scheduler**: ExponentialLR
- **Loss**: CrossEntropyLoss
- **Hardware**: automatic GPU usage when available; multi-worker DataLoader on CPU

## Expected Results

- **Validation accuracy**: 85–93%
- **Training time**: ~30–60 min (GPU) / 3–5 h (CPU)
- **Convergence**: typically between epochs 50–80

## Customization

Edit `config.py` to change hyperparameters:

```python
LEARNING_RATE = 0.0005  # Lower for more stable training
BATCH_SIZE = 32         # Higher if more VRAM is available
EPOCHS = 150            # More epochs for better convergence
```

Modify `model.py` to add/remove layers, change filter sizes, or adjust dropout rates.

## License

MIT — see [LICENSE](LICENSE).
