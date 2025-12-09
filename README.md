# G2Net Gravitational Wave Detection

> **Deep Learning Final Project - Ton Duc Thang University**
>
> **Topic:** Detecting Gravitational Waves in Noisy Signals using Deep Learning
>
> **Supervisor:** Assoc. Prof. Dr. Lê Anh Cường
>
> **Students:**
> - Đào Nguyễn Tấn Đạt (52300097)
> - Nguyễn Duy Minh Đăng (52300095)

---

## Project Results

Our solution utilizes an ensemble of **5 EfficientNet-B4 models** trained with **CQT Spectrograms**.

| Metric | Score | Description |
| :--- | :--- | :--- |
| **Private LB** | **0.86613** | ~ Rank 807 (Late Submission) |
| **Public LB** | 0.86731 | Tested on 16% of data |
| **CV Score** | 0.86405 | 5-Fold Stratified Cross-Validation |

---

## Solution Overview

We built an end-to-end Deep Learning pipeline designed to handle non-stationary noise in gravitational wave signals.

### 1. Signal Processing (1D)
* **Bandpass Filter:** Applied a [20Hz, 500Hz] filter to remove strong seismic noise (low freq) and quantum noise (high freq).
* **Spectral Whitening:** Normalized the signal energy by dividing it by the noise Power Spectral Density (PSD), effectively "flattening" the colored noise floor.

### 2. Feature Extraction (2D)
* **CQT (Constant Q-Transform):** Used `nnAudio` library for GPU-accelerated on-the-fly spectrogram generation.
* **Why CQT?** Unlike STFT (linear scale), CQT uses a **Logarithmic frequency scale**, providing high resolution at low frequencies (20-60Hz) where gravitational waves initiate.

### 3. Model Architecture
* **Backbone:** `tf_efficientnet_b4_ns` (Noisy Student pre-trained).
* **Input:** 3-channel CQT Spectrograms (LIGO Hanford, LIGO Livingston, Virgo).
* **Head:** Global Average Pooling $\rightarrow$ Dropout (0.2) $\rightarrow$ Linear $\rightarrow$ Sigmoid.

### 4. Training Strategy
* **Loss Function:** `BCEWithLogitsLoss` for numerical stability.
* **Optimizer:** `AdamW` with Weight Decay.
* **Scheduler:** `OneCycleLR` (Super-convergence).
* **Augmentation:**
    * **Time Shift:** Randomly rolling the signal to learn translation invariance.
    * **Mixup (alpha=0.2):** Blending samples to create soft labels, preventing the model from becoming over-confident on noisy data.

---

## Repository Structure

```text
├── solution.ipynb    # Main notebook containing the full Pipeline (Training + Inference)
├── submission.csv    # Final ensemble result (Private LB 0.866)
├── README.md         # Project documentation
└── requirements.txt  # List of dependencies
