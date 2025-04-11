(Currently Updating: May Include some issues and erros)

## Hyperspectral Imaging-Based Perception in Autonomous Driving: Semantic Segmentation Models

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-orange)](https://pytorch.org/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/imadalishah/HSI_SemanticSegmentationModels_AD_ADAS/blob/main/HSI_SemanticSegmentation_Models.ipynb)

This repository contains the implementation of various semantic segmentation models (SSMs) for hyperspectral imaging (HSI) data as described in our paper: [Hyperspectral Imaging-Based Perception in Autonomous Driving Scenarios: Benchmarking Baseline Semantic Segmentation Models](https://arxiv.org/pdf/2410.22101).

### Overview

Hyperspectral Imaging (HSI) offers significant advantages over traditional RGB imaging, that have been proven in the domains like remote sensing, agriculture, and medicine. Recently, it is increasingly being investigated for Advanced Driving Assistance Systems (ADAS) and Autonomous Driving (AD) scenarious. This repository provides a modular implementation of several baseline SSMs adapted for HSI (arbitrary dimensioned) data:

- **UNet**: Basic UNet architecture
- **UNet_CBAM**: UNet with Convolutional Block Attention Module
- **UNet_CA**: UNet with Coordinate Attention
- **UNet_CBAM_SkipConnection**: UNet with CBAM applied to skip connections
- **DeepLabV3Plus**: DeepLabV3+ architecture
- **HRNet**: High-Resolution Network
- **PSPNet**: Pyramid Scene Parsing Network

### Key Features

- Modular architecture with easy component swapping
- Support for various activation functions
- Factory pattern for model instantiation
- Flexible input dimensions handling
- Attention mechanisms integration (CBAM, Coordinate Attention)
- Comprehensive testing framework

## Installation

```bash
git clone https://github.com/imadalishah/HSI_SemanticSegmentationModels_AD_ADAS.git
cd HSI_SemanticSegmentationModels_AD_ADAS
pip install -r requirements.txt # Will upload shortly
```

## Usage

### Model Creation

```py
import torch
from models import create_model

# Create a UNet model
model = create_model(
    model_type='UNet',
    in_channels=15,  # Number of input spectral bands
    out_channels=10,  # Number of output classes
    features=[64, 128, 256, 512],  # Feature dimensions
    act='ReLU'  # Activation function
)

# Process a test input tensor for verification
input_tensor = torch.randn(1, 15, 210, 150)  # [batch_size, channels, height, width]
output = model(input_tensor)  # Shape: [1, 10, 210, 150]
```

### Available Models

```py
# UNet with CBAM attention
model_cbam = create_model('UNet_CBAM', in_channels=15, out_channels=10)

# UNet with Coordinate Attention
model_ca = create_model('UNet_CA', in_channels=15, out_channels=10)

# UNet with CBAM on skip connections
model_cbam_skip = create_model('UNet_CBAM_SkipConnection', in_channels=15, out_channels=10)

# DeepLabV3+
model_deeplab = create_model('DeepLabV3Plus', in_channels=15, out_channels=10)

# HRNet
model_hrnet = create_model('HRNet', in_channels=15, out_channels=10)

# PSPNet
model_pspnet = create_model('PSPNet', in_channels=15, out_channels=10)
```

## Paper Abstract

Hyperspectral Imaging (HSI) is known for its advantages over traditional RGB imaging in remote sensing, agriculture, and medicine. Recently, it has gained attention for enhancing Advanced Driving Assistance Systems (ADAS) perception. Several HSI datasets such as HyKo, HSI-Drive, HSI-Road, and Hyperspectral City have been made available. However, a comprehensive evaluation of semantic segmentation models (SSM) using these datasets is lacking.

To address this gap, we evaluated the available annotated HSI datasets on four deep learning-based baseline SSMs: DeepLab v3+, HRNet, PSPNet, and U-Net, along with its two variants: Coordinate Attention (UNet-CA) and Convolutional Block-Attention Module (UNet-CBAM). The original model architectures were adapted to handle the varying spatial and spectral dimensions of the datasets.

Our results indicate that UNet-CBAM, which extracts channel-wise features, outperforms other SSMs and shows potential to leverage spectral information for enhanced semantic segmentation. This study establishes a baseline SSM benchmark on available annotated datasets for future evaluation of HSI-based ADAS perception.

## Model Architecture

The implementation follows a modular approach with a factory pattern for model creation:

```py
def create_model(model_type, in_channels, out_channels, features=[64, 128, 256, 512], act='ReLU'):
    # Factory function that returns the requested model type
    if model_type == 'UNet':
        return BaseUNet(in_channels, out_channels, features, act)
    elif model_type == 'UNet_CBAM':
        return BaseUNet(in_channels, out_channels, features, act,
                        attention_module=CBAM, attention_params={'reduction': 16})
    # ... others: Deeplabv3+, ...
```

## Citation

If you use this code in your research, please cite our paper and also the relevant authors of the used Models:

```bibtex
@article{shah2024hyperspectral,
  title={Hyperspectral Imaging-Based Perception in Autonomous Driving Scenarios: Benchmarking Baseline Semantic Segmentation Models},
  author={Shah, Imad Ali and Li, Jiarong and Glavin, Martin and Jones, Edward and Ward, Enda and Deegan, Brian},
  journal={arXiv preprint arXiv:2410.22101},
  year={2024}
}
```

## Acknowledgments

- This research was presented at [IEEE WHISPERS 2024](https://doi.org/10.1109/WHISPERS65427.2024.10876494)
- In addition to all referenced research in the paper, we thank the authors of the HSI datasets (i.e. HyKo, HSI-Drive, HSI-Road, and Hyperspectral City) and SSMs (i.e. UNet, PSPNet, HRNet, DeepLabv3+, CBAM and CA)

## Contact

For questions or feedback, please open an issue or contact the authors.
