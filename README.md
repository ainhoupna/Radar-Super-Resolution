# Deep Learning-based Super-Resolution for Meteorological Radar Imagery

**Authors:** Ainhoa Del Rey and Iñigo Goikoetxea  
**Date:** January 2026

## Overview

This project implements a **Single Image Super-Resolution (SISR)** pipeline to enhance the spatial resolution of meteorological radar reflectivity data. By leveraging deep learning techniques, specifically a **Second-order Attention Network (SAN)**, we upscale radar imagery from its native 1 km resolution to 500m ($\times 2$) and 250m ($\times 4$).

The goal is to reconstruct fine-grained convective structures and reflectivity gradients that are often lost in standard radar products, thereby improving the accuracy of nowcasting and hydrological modeling.

## Key Features

-   **Advanced Architecture**: Implementation of **SAN (Second-order Attention Network)** for superior feature correlation modeling compared to standard EDSR.
-   **Multi-Scale Support**: Dedicated models for **$\times 2$** and **$\times 4$** upscaling.
-   **Meteorological Optimization**:
    -   **Max Pooling** downsampling to preserve peak intensity (dBZ).
    -   **SSIM Loss** integration to maintain structural integrity of storm cells.
    -   **Dihedral Augmentation** to handle diverse storm orientations.
-   **High-Performance Training**: Automatic Mixed Precision (AMP) support for efficient training on limited hardware.
-   **Georeferenced Visualization**: Pipeline to overlay super-resolved predictions onto OpenStreetMap (OSM) layers for real-world context.
-   **Experiment Tracking**: Full integration with **MLflow** for logging metrics, parameters, and artifacts.

## Results

Our SAN model significantly outperforms baseline bicubic interpolation and standard EDSR variants.

| Experiment | Scale | PSNR (dB) | SSIM |
| :--- | :---: | :---: | :---: |
| Bicubic Baseline | $\times 2$ | 29.06 | 0.8474 |
| EDSR (Best Variant) | $\times 2$ | 32.16 | 0.9023 |
| **SAN (Ours)** | **$\times 2$** | **32.57** | **0.9195** |

*Qualitative results show sharper convective cores and better-defined storm boundaries.*

## Installation

1.  **Clone the repository**:
    ```bash
    git clone <repository_url>
    cd Radar-Super-Resolution
    ```

2.  **Install dependencies**:
    Ensure you have Python 3.8+ and PyTorch installed with CUDA support.
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip install -r requirements.txt
    ```
    *(Note: If `requirements.txt` is missing, key dependencies include `torch`, `numpy`, `Pillow`, `tqdm`, `mlflow`, `safetensors`, `piq`, `scikit-image`).*

## Usage

### 1. Training
To train the SAN model for $\times 2$ upscaling with SSIM loss and Augmentation:
```bash
python "train_san_x2 ssim_amp_aug_.py"
```
For $\times 4$ upscaling:
```bash
python "train_san_x4 ssim_amp_aug_.py"
```

### 2. Inference (Single Image)
Run the prediction script to super-resolve a specific radar image:
```bash
python predict_san.py
```
*Configuration (input path, checkpoint path) can be modified directly in the `predict_san.py` script.*

### 3. Evaluation
To evaluate the trained models on the test set and calculate PSNR/SSIM metrics:
```bash
python evaluate_san.py
```

### 4. Visualization
To generate georeferenced visualizations (requires configured checkpoints):
```bash
python georef_san_dana.py
```

## Project Structure

-   `models/`: Contains the SAN and EDSR model definitions.
-   `dataset_split/`: Directory for Train/Val/Test datasets (using `.safetensors`).
-   `checkpoints_san/`: Saved model weights.
-   `final_report.tex`: Detailed LaTeX report of the project.
-   `mlruns/`: MLflow experiment logs.

## License

This project is developed for academic purposes. Data provided by AEMET.
