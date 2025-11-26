# G2Net Gravitational Wave Detection Project

> **Final Project for Deep Learning Course**

This project implements a Deep Learning solution for the **G2Net Gravitational Wave Detection** challenge. It utilizes **Constant Q-Transform (CQT)** for signal-to-image conversion and an **EfficientNet** backbone to detect gravitational wave signals (binary classification) in noisy time-series data from LIGO/Virgo interferometers.

## Results

| Model | Image Size | Training Epochs | Best CV AUC | Public LB Score | Private LB Score |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **EfficientNet-B4** | 64x69 (CQT) | 12 | **0.86405** | **0.86620** | **0.86491** |

## Project Structure

- **`solution.ipynb`**: The main notebook containing the training loop, model definition, validation, and inference logic.
- **`utils.py`**: Helper script for signal processing (Whitening, Bandpass), dataset loading, and configuration.
- **`requirements.txt`**: List of Python dependencies.
- **`submission.csv`**: The final prediction file generated for Kaggle submission.

## Key Features

- **Signal Processing**:
  - **CQT (Constant Q-Transform)**: Converts 1D waves to 2D spectrograms using `nnAudio` (GPU-accelerated).
  - **Whitening & Bandpass**: Applied to remove noise and normalize the signal (20-500Hz range).
- **Model Architecture**:
  - **Backbone**: `tf_efficientnet_b4_ns` (via `timm`).
  - **Input**: 3 Channels (LIGO Hanford, LIGO Livingston, Virgo).
- **Training Strategy**:
  - **Stratified K-Fold**: 5-fold cross-validation.
  - **Mixup Augmentation**: Improves model generalization.
  - **Optimizer**: AdamW with Cosine Annealing LR.

## Model Weights

Since the trained model file is large (>70MB), it is hosted externally. Please download the weights to run inference:

[Download Model Weights](https://drive.google.com/file/d/137w0EIUFuoUWl8Cm1ZEUoajohLOYDuOM/view?usp=sharing)

After downloading, place the `best_model_fold_0.pth` file in the root directory.

## Installation

1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
3. Download the Dataset from [Kaggle G2Net Competition](https://www.kaggle.com/c/g2net-gravitational-wave-detection/data) and extract it.

## Usage

1. **Configuration**:
   - The configuration is centralized in `utils.py` (class `ProjectConfig`).
   - Please update `DATA_DIR` to point to your local dataset path before running.

2. **Run Training / Inference**:
   - Open `solution.ipynb` using Jupyter Lab or Notebook:
     ```bash
     jupyter notebook solution.ipynb
     ```
   - Run all cells to reproduce the training process and generate `submission.csv`.
