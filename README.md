## Hyperspectral Imaging-Based Perception in Autonomous Driving: Semantic Segmentation Models
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-orange)](https://pytorch.org/)

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
git clone https://github.com/yourusername/HSI-Segmentation.git
cd HSI-Segmentation
pip install -r requirements.txt
```

## Usage

### Model Creation

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

# Process an input tensor
input_tensor = torch.randn(1, 15, 210, 150)  # [batch_size, channels, height, width]
output = model(input_tensor)  # Shape: [1, 10, 210, 150]

### Available Models

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

## Model Testing

The repository includes a testing framework to verify model functionality with different input dimensions:

from test_models import test_all_models

# Test dimensions
dimensions = [
    [15, 210, 150],   # [channels, height, width]
    [128, 472, 355],
    [25, 209, 416]
]

# Run tests
test_all_models(dimensions)

## Paper Abstract

Hyperspectral Imaging (HSI) is known for its advantages over traditional RGB imaging in remote sensing, agriculture, and medicine. Recently, it has gained attention for enhancing Advanced Driving Assistance Systems (ADAS) perception. Several HSI datasets such as HyKo, HSI-Drive, HSI-Road, and Hyperspectral City have been made available. However, a comprehensive evaluation of semantic segmentation models (SSM) using these datasets is lacking.

To address this gap, we evaluated the available annotated HSI datasets on four deep learning-based baseline SSMs: DeepLab v3+, HRNet, PSPNet, and U-Net, along with its two variants: Coordinate Attention (UNet-CA) and Convolutional Block-Attention Module (UNet-CBAM). The original model architectures were adapted to handle the varying spatial and spectral dimensions of the datasets.

Our results indicate that UNet-CBAM, which extracts channel-wise features, outperforms other SSMs and shows potential to leverage spectral information for enhanced semantic segmentation. This study establishes a baseline SSM benchmark on available annotated datasets for future evaluation of HSI-based ADAS perception.

## Model Architecture

The implementation follows a modular approach with a factory pattern for model creation:

def create_model(model_type, in_channels, out_channels, features=[64, 128, 256, 512], act='ReLU'):
    # Factory function that returns the requested model type
    if model_type == 'UNet':
        return BaseUNet(in_channels, out_channels, features, act)
    elif model_type == 'UNet_CBAM':
        return BaseUNet(in_channels, out_channels, features, act,
                        attention_module=CBAM, attention_params={'reduction': 16})
    # ... other model types

## Results

Our testing confirms that all models successfully handle various input dimensions:

```
Testing input with dimensions: [C=15, H=210, W=150]
  Testing UNet...
    Success! Output shape: torch.Size([1, 10, 210, 150])
  Testing UNet_CBAM...
    Success! Output shape: torch.Size([1, 10, 210, 150])
  Testing UNet_CA...
    Success! Output shape: torch.Size([1, 10, 210, 150])
  Testing UNet_CBAM_SkipConnection...
    Success! Output shape: torch.Size([1, 10, 210, 150])
  Testing DeepLabV3Plus...
    Success! Output shape: torch.Size([1, 10, 210, 150])
  Testing HRNet...
    Success! Output shape: torch.Size([1, 10, 210, 150])
  Testing PSPNet...
    Success! Output shape: torch.Size([1, 10, 210, 150])
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

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This research was presented at IEEE WHISPERS 2024
- We thank the authors of the HSI datasets: HyKo, HSI-Drive, HSI-Road, and Hyperspectral City

## Contact

For questions or feedback, please open an issue or contact the authors.
