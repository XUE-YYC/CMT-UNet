# CMT-UNet: CNN-Mamba Transformer UNet for Semantic Segmentation

## Overview

CMT-UNet is a state-of-the-art hybrid architecture for semantic segmentation that combines the strengths of Convolutional Neural Networks (CNNs), Vision Mamba (VSSM), and Transformer-based attention mechanisms. This novel approach achieves superior performance on remote sensing segmentation benchmarks like Vaihingen, Urban, and UAVid datasets by integrating:

1. **ResNet backbone** for robust local feature extraction
2. **Vision Mamba (VSSM)** for efficient global context modeling with linear complexity
3. **Hybrid attention mechanisms** for multi-scale feature fusion
4. **Wavelet transform convolutions** for multi-resolution feature processing



## Key Features

- **Hybrid CNN-Mamba-Transformer architecture** combines complementary strengths of different paradigms
- **Efficient global context modeling** with Vision Mamba (VSSM) encoder
- **Wavelet-based convolutions** for enhanced feature extraction
- **Multi-scale feature fusion** with attention mechanisms
- **Pretrained backbone support** for accelerated convergence
- **Multi-dataset support** (Vaihingen, Urban, UAVid)
- **Parallel processing** for efficient training

## Installation

1. Clone the repository:

```bash
git clone https://github.com/your_username/CMT-UNet.git
cd CMT-UNet
```

2. Create and activate a conda environment (recommended):

```bash
conda create -n cmt_unet python=3.8
conda activate cmt_unet
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Install Mamba components:

```bash
pip install causal-conv1d==1.2.0 mamba-ssm==1.2.0
```

## Dataset Preparation

### Supported Datasets

1. **Vaihingen** 
2. **Urban**
3. **UAVid**

### Preparation Steps

1. Download datasets from their official sources

2. Organize directory structure:

   ```
   /path/to/dataset/
   ├── train/
   │   ├── images/
   │   └── masks/
   └── test/
       ├── images/
       └── masks/
   ```

3. Preprocess data using the cropping script:

```bash
python test2.py --dataset_root /path/to/dataset
```

## Training

### Configuration

Edit `utils_Mamba.py` to configure:

```python
MODEL = 'CMT_UNet'       # Model architecture
DATASET = 'Vaihingen'    # Dataset name
BATCH_SIZE = 8           # Batch size
WINDOW_SIZE = (256, 256) # Training patch size
```

### Start Training

```bash
python train_Mamba.py
```

### Training Features

- Automatic mixed precision
- Multi-step learning rate scheduling
- Class-balanced loss weighting
- Model checkpointing
- Multi-GPU support

## Testing

### Evaluate Model

```bash
python train_Mamba.py --mode Test --checkpoint /path/to/checkpoint.pth
```

### Evaluation Outputs

1. Segmentation masks
2. Quantitative metrics (mIoU, Accuracy, F1 Score)
3. Visual predictions

## Customization

### Model Architecture

Modify `CMT_UNet.py` to:

- Adjust decoder channels
- Change attention mechanisms
- Modify fusion blocks
- Customize wavelet transform parameters

### Training Parameters

Edit `utils_Mamba.py` to:

- Adjust batch size
- Modify learning rate schedule
- Change class weights
- Set window size and stride

## Code Structure

```
CMT-UNet/
├── test2.py                # Data preprocessing and cropping
├── MSAA.py                 # Multi-Scale Attention Module
├── utils_Mamba.py          # Configuration and utilities
├── UNetFormer.py           # UNetFormer model implementation
├── CMT_UNet.py             # CMT-UNet model implementation
├── train_Mamba.py          # Training and testing script
├── wtconv2d.py             # Wavelet transform convolutions
├── SwinUMamba.py           # Vision Mamba (VSSM) implementation
├── requirements.txt        # Dependencies
└── README.md               # This document
```

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{cmtunet2024,
  title={CMT-UNet: CNN-Mamba Transformer UNet for Semantic Segmentation},
  author={Author Name},
  year={2024},
  howpublished={\url{https://github.com/your_username/CMT-UNet}}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

This implementation builds upon:

- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)
- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- [Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model](https://arxiv.org/abs/2401.09417)
