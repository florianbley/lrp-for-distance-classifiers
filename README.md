# Neuralized SVM and KNN: Fast and Accurate Explanations of Distance-Based Classifiers

This repository demonstrates how to make Support Vector Machines (SVMs) and K-Nearest Neighbors (KNN) classifiers explainable using **neuralization**.

## What is Neuralization?

Neuralization transforms traditional machine learning models into a form that resembles neural networks. This enables the use of relevance propagation-based explainers.

### The Key Idea

An RBF SVM's decision function:
```
f(x) = Σᵢ αᵢ · exp(-γ||x - xᵢ||²) + b
```

Can be rewritten as a difference of log-sum-exp terms:
```
g(x) = log Σᵢ exp(-γ·zᵢ⁺) - log Σⱼ exp(-γ·zⱼ⁻)
```

This form can be further expressed as a two-layer neural network, allowing us to apply relevance propagation-based explanation methods.

## Repository Contents

```
upload_to_repo/
├── neuralised_svm.py          # Neuralized SVM implementation (heavily commented)
├── neuralised_knn.py          # Neuralized KNN implementation (heavily commented)
├── demo_2d_visualization.py   # 2D visualization comparing original vs neuralized
├── demo_mnist.py              # Full MNIST demo with explanations
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── figures/                   # Output directory for visualizations
```

## Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

### Run the 2D Demo (Recommended First)

```bash
python demo_2d_visualization.py
```

This creates a visualization showing that the neuralized SVM produces identical decision boundaries to the original SVM, even though the raw output values differ.

### Run the MNIST Demo

```bash
python demo_mnist.py
```

This script:
1. Loads MNIST digits 4 and 9
2. Applies distance normalization preprocessing
3. Trains SVM and KNN with hyperparameter tuning
4. Verifies neuralization preserves classifications
5. Generates heatmap explanations

## Explanation Methods

The neuralized models support three explanation rules:

### 1. GI (Gradient × Input)
- Uses the gradient of the neuralized function
- Works well when data is centered around zero
- Formula: `R = x · 2 · (mean_pos_sv - mean_neg_sv)`

### 2. Midpoint Rule
- Uses squared differences to support vectors
- More stable for non-standardized data
- Formula: `R = weighted_sq_diff_neg - weighted_sq_diff_pos`

### 3. Hybrid Rule (Recommended)
- Combines GI and Midpoint: `R = (1-η)·GI + η·Midpoint`
- η (eta) controls the balance
- Heuristic: η depends on SVM gamma (or fixed at 0.8 for KNN)

## Key Parameters

### Distance Normalization
Before training, data is normalized so that the median pairwise squared distance equals 1, ensuring interpretability of the degree of non-linearity with gamma.

### Eta Heuristic (SVM)
```python
if log10(gamma) < -0.5:
    eta = 0.0      # Pure GI (smooth kernel)
elif log10(gamma) > 1.5:
    eta = 1.0      # Pure Midpoint (sharp kernel)
else:
    eta = log10(gamma) * 0.4 + 0.4  # Linear interpolation
```

### Eta Heuristic (KNN)
```python
eta = 0.8  # Fixed value (empirically determined)
```

## Interpreting Explanations

The explanation heatmaps show feature importance:
- **Positive values (red)**: Feature contributes toward the positive class
- **Negative values (blue)**: Feature contributes toward the negative class
- **Zero (white)**: Feature is neutral

## Understanding the Code

Key methods:

### `NeuralisedSVM`
- `__init__`: Extracts support vectors and dual coefficients from sklearn SVC
- `compute_z`: Computes distance-based terms to support vectors
- `forward`: The neuralized decision function (log-sum-exp form)
- `compute_point_pair_weights`: Softmax attention over support vectors
- `explain`: Implements GI, Midpoint, and Hybrid explanation rules

### `NeuralisedKNN`
- Inherits from `NeuralisedSVM`
- Uses uniform weights over k-nearest neighbors instead of softmax
- Same `explain` method works for both models

## Citation

If you use this code in your research, please cite:

```bibtex
@article{Bley2025,
  title={Fast and Accurate Explanations of Distance-Based Classifiers
         by Uncovering Latent Explanatory Structures},
  author={Bley, Florian and Kauffmann, Jacob and Krug, Simon Le{\'o}n and
          M{\"u}ller, Klaus-Robert and Montavon, Gr{\'e}goire},
  journal={arXiv preprint arXiv:2508.03913},
  year={2025}
}
```

Paper: [arXiv:2508.03913](https://arxiv.org/abs/2508.03913)
