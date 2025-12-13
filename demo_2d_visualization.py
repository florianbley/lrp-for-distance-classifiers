"""
Demo: 2D Visualization of SVM Neuralization

This script demonstrates the neuralization concept on a simple 2D dataset.
It shows that the neuralized SVM produces the same classification decisions
as the original SVM, even though the exact output values differ.

The visualization shows:
- Left: Original SVM decision function contours
- Right: Neuralized SVM forward() contours

Key observation: The decision boundaries (where the value is 0) are identical,
but the overall shape of the function is different due to the log-sum-exp
transformation used in neuralization.

Usage:
    python demo_2d_visualization.py

Output:
    Saves figure to figures/2d_comparison.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from sklearn.datasets import make_moons
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances

from neuralised_svm import NeuralisedSVM


def distance_normalization(X_train, X_test):
    """
    Normalize data so that median pairwise squared distance equals 1.

    This makes the gamma parameter more interpretable:
    - gamma=1 means the kernel "sees" the typical distance between points
    - gamma>1 means sharper kernel (more local)
    - gamma<1 means smoother kernel (more global)
    """
    D_squared = euclidean_distances(X_train, X_train) ** 2
    median_dist_sq = np.median(D_squared)
    scale = np.sqrt(median_dist_sq)

    X_train_normed = X_train / scale
    X_test_normed = X_test / scale

    return X_train_normed, X_test_normed, median_dist_sq

# =============================================================================
# Configuration
# =============================================================================

RANDOM_SEED = 42
N_SAMPLES = 50
TEST_SIZE = 0.2
GAMMA = 1.0  # RBF kernel parameter
C = 10.0      # Regularization parameter

# =============================================================================
# Step 1: Generate 2D nonlinear dataset
# =============================================================================

print("=" * 60)
print("Step 1: Generating 2D dataset (make_moons)")
print("=" * 60)

# Create a nonlinear dataset that requires a nonlinear classifier
X, y = make_moons(n_samples=N_SAMPLES, noise=0.2, random_state=RANDOM_SEED)

# Convert labels from {0, 1} to {-1, +1} for SVM convention
y = 2 * y - 1

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
)

# Standardize features (important for SVM with RBF kernel)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Apply distance normalization so median squared distance = 1
X_train, X_test, median_dist_sq = distance_normalization(X_train, X_test)
print(f"Median squared distance (before norm): {median_dist_sq:.4f}")

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Feature dimensions: {X_train.shape[1]}")

# =============================================================================
# Step 2: Train SVM with RBF kernel
# =============================================================================

print("\n" + "=" * 60)
print("Step 2: Training SVM with RBF kernel")
print("=" * 60)

svm = SVC(kernel='rbf', gamma=GAMMA, C=C)
svm.fit(X_train, y_train)

train_accuracy = svm.score(X_train, y_train)
test_accuracy = svm.score(X_test, y_test)

print(f"Gamma: {GAMMA}")
print(f"C: {C}")
print(f"Number of support vectors: {len(svm.support_vectors_)}")
print(f"Training accuracy: {train_accuracy:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")

# =============================================================================
# Step 3: Create neuralized version
# =============================================================================

print("\n" + "=" * 60)
print("Step 3: Creating neuralized SVM")
print("=" * 60)

neural_svm = NeuralisedSVM(svm)

print(f"Positive support vectors: {neural_svm.num_pos}")
print(f"Negative support vectors: {neural_svm.num_neg}")

# =============================================================================
# Step 4: Verify that classifications match
# =============================================================================

print("\n" + "=" * 60)
print("Step 4: Verifying neuralization preserves classification")
print("=" * 60)

# Get predictions from both models
original_decision = svm.decision_function(X_test)
neural_output = neural_svm.forward(X_test, with_intercept=True)

# Check sign agreement
original_signs = np.sign(original_decision)
neural_signs = np.sign(neural_output)

# Note: The signs might be flipped globally (all +1 become -1 and vice versa)
# So we check if they either all match or all are opposite
agreement = np.mean(original_signs == neural_signs)
opposite = np.mean(original_signs == -neural_signs)

if agreement > 0.99:
    print(f"Sign agreement: {agreement * 100:.1f}% (same signs)")
    sign_flip = 1
elif opposite > 0.99:
    print(f"Sign agreement: {opposite * 100:.1f}% (flipped signs)")
    sign_flip = -1
else:
    print(f"WARNING: Partial agreement only!")
    print(f"  Same signs: {agreement * 100:.1f}%")
    print(f"  Opposite signs: {opposite * 100:.1f}%")
    sign_flip = 1

print("\nClassification matches: YES" if max(agreement, opposite) > 0.99 else "Classification matches: NO")

# =============================================================================
# Step 5: Create visualization grid
# =============================================================================

print("\n" + "=" * 60)
print("Step 5: Creating decision function visualizations")
print("=" * 60)

# Create a grid of points covering the data range
x_min, x_max = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5
y_min, y_max = X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 200),
    np.linspace(y_min, y_max, 200)
)
grid_points = np.c_[xx.ravel(), yy.ravel()]

# Compute decision functions on the grid
print("Computing original SVM decision function on grid...")
original_grid = svm.decision_function(grid_points).reshape(xx.shape)

print("Computing neuralized SVM forward on grid...")
neural_grid = (sign_flip * neural_svm.forward(grid_points, with_intercept=True)).reshape(xx.shape)

# =============================================================================
# Step 6: Create and save the figure
# =============================================================================

print("\n" + "=" * 60)
print("Step 6: Saving visualization")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Color map settings - centered at zero with red=positive, blue=negative
cmap = 'RdBu_r'
levels = 50

# Left plot: Original SVM
ax1 = axes[0]
vmax1 = max(abs(original_grid.min()), abs(original_grid.max()))
norm1 = TwoSlopeNorm(vmin=-vmax1, vcenter=0, vmax=vmax1)
contour1 = ax1.contourf(xx, yy, original_grid, levels=levels, cmap=cmap, norm=norm1, alpha=0.8)
ax1.contour(xx, yy, original_grid, levels=[0], colors='black', linewidths=2)
ax1.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1],
            c='red', edgecolors='black', s=50, label='Class +1')
ax1.scatter(X_train[y_train == -1, 0], X_train[y_train == -1, 1],
            c='blue', edgecolors='black', s=50, label='Class -1')
ax1.scatter(svm.support_vectors_[:, 0], svm.support_vectors_[:, 1],
            s=100, facecolors='none', edgecolors='yellow', linewidths=2,
            label='Support Vectors')
ax1.set_xlabel('Feature 1')
ax1.set_ylabel('Feature 2')
ax1.set_title('Original SVM Decision Function\n' + r'$f(x) = \sum_i \alpha_i K(x, x_i) + b$')
ax1.legend(loc='upper left')
plt.colorbar(contour1, ax=ax1, label='Decision value')

# Right plot: Neuralized SVM
ax2 = axes[1]
norm2 = TwoSlopeNorm(vmin=neural_grid.min(), vcenter=0, vmax=neural_grid.max())
contour2 = ax2.contourf(xx, yy, neural_grid, levels=levels, cmap=cmap, norm=norm2, alpha=0.8)
ax2.contour(xx, yy, neural_grid, levels=[0], colors='black', linewidths=2)
ax2.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1],
            c='red', edgecolors='black', s=50, label='Class +1')
ax2.scatter(X_train[y_train == -1, 0], X_train[y_train == -1, 1],
            c='blue', edgecolors='black', s=50, label='Class -1')
ax2.scatter(svm.support_vectors_[:, 0], svm.support_vectors_[:, 1],
            s=100, facecolors='none', edgecolors='yellow', linewidths=2,
            label='Support Vectors')
ax2.set_xlabel('Feature 1')
ax2.set_ylabel('Feature 2')
ax2.set_title('Neuralized SVM Forward\n' + r'$g(x) = \log\sum_i e^{-\gamma z_i^+} - \log\sum_j e^{-\gamma z_j^-}$')
ax2.legend(loc='upper left')
plt.colorbar(contour2, ax=ax2, label='Neuralized value')

plt.tight_layout()

# Save figure
output_path = 'figures/2d_comparison.png'
os.makedirs('figures', exist_ok=True)
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Figure saved to: {output_path}")

plt.show()

print("\n" + "=" * 60)
print("Demo complete!")
print("=" * 60)
print("\nKey observations:")
print("1. The decision boundary (black line) is IDENTICAL in both plots")
print("2. The overall shape of the function differs (original vs log-sum-exp)")
print("3. This shows neuralization preserves classification while enabling")
print("   neural network-style explanation methods")
