# Deep Learning-based Super-Resolution for Meteorological Radar Imagery

<p align="center">
  <strong>Enhancing radar reflectivity data from 1km to 250m resolution using ESDR and Second-order Attention Networks</strong>
</p>

<p align="center">
  <b>Authors:</b> Ainhoa Del Rey & Iñigo Goikoetxea<br>
  <b>Date:</b> January 2026
</p>

---

## 📋 Overview

This project implements a **Single Image Super-Resolution (SISR)** pipeline specifically designed to enhance the spatial resolution of meteorological radar reflectivity data. By leveraging deep learning techniques, specifically a **Second-order Attention Network (SAN)**, we upscale radar imagery from its native **1 km** resolution to:

- **500m** (×2 upscaling)
- **250m** (×4 upscaling)

### Why Super-Resolution for Radar?

Standard radar products often lose fine-grained details of convective structures and reflectivity gradients. This project aims to:

- 🎯 Reconstruct sharp convective cores and storm cell boundaries
- 📊 Improve accuracy for nowcasting applications
- 💧 Enhance input quality for hydrological modeling
- 🗺️ Provide better georeferenced visualizations for emergency response

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| **Advanced Architecture** | Implementation of SAN (Second-order Attention Network) for superior feature correlation modeling |
| **Multi-Scale Support** | Dedicated models for ×2 and ×4 upscaling with scale-specific optimizations |
| **Max Pooling Downsampling** | Preserves peak intensity (dBZ) values during training data generation |
| **SSIM Loss Integration** | Maintains structural integrity of storm cells and precipitation patterns |
| **Dihedral Augmentation** | Handles diverse storm orientations with 8-fold augmentation |
| **Mixed Precision Training** | AMP (Automatic Mixed Precision) support for efficient training on limited hardware |
| **Georeferenced Visualization** | Pipeline to overlay super-resolved predictions onto OpenStreetMap layers |
| **Experiment Tracking** | Full integration with MLflow for logging metrics, parameters, and artifacts |

---

## 📊 Results

Our SAN model significantly outperforms baseline methods:

### ×2 Upscaling (1km → 500m)

| Model | PSNR (dB) ↑ | SSIM ↑ |
|:------|:-----------:|:------:|
| Bicubic Interpolation | 29.06 | 0.8474 |
| EDSR (32 blocks) | 32.16 | 0.9023 |
| **SAN (Ours)** | **32.57** | **0.9195** |

> 💡 *Qualitative results show sharper convective cores and better-defined storm boundaries compared to bicubic interpolation.*

---

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- PyTorch with CUDA support

### Setup

1. **Clone the repository**:
   ```bash
   git clone <repository_url>
   cd Radar-Super-Resolution
   ```

2. **Install PyTorch** (with CUDA 11.8):
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   <details>
   <summary>📦 Key Dependencies (if requirements.txt is missing)</summary>
   
   ```
   torch
   torchvision
   numpy
   Pillow
   tqdm
   mlflow
   safetensors
   piq
   scikit-image
   pyproj
   rasterio
   contextily
   matplotlib
   ```
   </details>

---

## 🚀 Usage

### 1. Training

Train the SAN model with SSIM loss and data augmentation:

```bash
# For ×2 upscaling (1km → 500m)
python "train_san_x2 ssim_amp_aug_.py"

# For ×4 upscaling (1km → 250m)
python "train_san_x4 ssim_amp_aug_.py"
```

#### Training Scripts Overview

| Script | Scale | Features |
|--------|:-----:|----------|
| `train_esdr.py` | ×4 | Baseline EDSR training |
| `train_esdr_x2.py` | ×2 | EDSR with L1 loss |
| `train_esdr_x2 ssim_amp_aug.py` | ×2 | EDSR + SSIM + AMP + Augmentation |
| `train_san_x2 ssim_amp_aug_.py` | ×2 | **SAN** + SSIM + AMP + Augmentation |
| `train_san_x4 ssim_amp_aug_.py` | ×4 | **SAN** + SSIM + AMP + Augmentation |

### 2. Inference (Single Image)

Run super-resolution on a specific radar image:

```bash
python predict_san.py
```

> ⚙️ Configuration (input path, checkpoint path) can be modified directly in the script.

### 3. Batch Inference

Process multiple images in a directory:

```bash
python predict_batch_san.py
```

### 4. Evaluation

Evaluate trained models on the test set and calculate PSNR/SSIM metrics:

```bash
# Evaluate SAN model
python evaluate_san.py

# Compare EDSR vs Bicubic baseline
python evaluate_edsr_vs_bicubic.py
```

### 5. Georeferenced Visualization

Generate georeferenced visualizations overlaid on OpenStreetMap:

```bash
python georef_san_dana.py
```

This produces comparison images showing the original radar data alongside the super-resolved output with geographic context.

---

## 📁 Project Structure

```
Radar-Super-Resolution/
├── models/
│   └── san/                    # SAN model architecture
│       ├── san.py              # Second-order Attention Network
│       ├── common.py           # Shared modules
│       └── MPNCOV/             # Matrix Power Normalized Covariance
├── dataset_split/              # Train/Val/Test datasets (.safetensors)
│   ├── train/
│   ├── val/
│   └── test/
├── checkpoints_san/            # Saved SAN model weights
├── mlruns/                     # MLflow experiment logs
├── train_*.py                  # Training scripts
├── predict_*.py                # Inference scripts
├── evaluate_*.py               # Evaluation scripts
├── georef_*.py                 # Georeferenced visualization
└── README.md
```

---

## 🧠 Model Architecture

### Second-order Attention Network (SAN)

The SAN architecture captures long-range dependencies through:

1. **Non-local Enhanced Residual Groups (NLRGs)**: Capture second-order feature statistics
2. **Second-order Channel Attention (SOCA)**: Model channel-wise interdependencies
3. **Region-level Non-local Module**: Capture spatial correlations

```
Input (LR) → Head → [NLRG × N] → Non-local → Tail → Upsampler → Output (SR)
```

### Configuration

| Parameter | ×2 Model | ×4 Model |
|-----------|:--------:|:--------:|
| Residual Groups | 10 | 10 |
| Residual Blocks | 10 | 10 |
| Feature Channels | 64 | 64 |
| Patch Size | 64 | 32 |
| Loss Function | 0.8×L1 + 0.2×SSIM | 0.8×L1 + 0.2×SSIM |

---

## 📈 Experiment Tracking with MLflow

All experiments are logged to MLflow for reproducibility:

```bash
# Start MLflow UI
mlflow ui --backend-store-uri sqlite:///mlflow.db

# View experiments at http://localhost:5000
```

Tracked metrics include:
- Training and validation loss
- PSNR and SSIM per epoch
- Best checkpoint paths
- Hyperparameters

---

## 📚 Data Format

Training data uses the **SafeTensors** format for efficient loading:

```python
# Each .safetensors file contains:
{
    'HR': tensor([1, 480, 480]),      # High-resolution ground truth
    'LR_2x': tensor([1, 240, 240]),   # Low-resolution (×2 downsampled)
    'LR_4x': tensor([1, 120, 120])    # Low-resolution (×4 downsampled)
}
```

Data preprocessing:
- Values normalized to [0, 1]
- Max pooling used for downsampling (preserves peak dBZ values)
- Dihedral augmentation (rotations + flips) during training
