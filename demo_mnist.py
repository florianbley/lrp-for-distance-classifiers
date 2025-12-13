"""
Demo: MNIST Digit Classification with Neuralized SVM and KNN Explanations

This script demonstrates the complete workflow of neuralized explanations:
1. Load MNIST data (digits 4 vs 9)
2. Apply distance normalization preprocessing
3. Train SVM and KNN classifiers with hyperparameter tuning
4. Verify that neuralized models produce same classifications
5. Compute explanations using the hybrid rule
6. Visualize explanations as heatmaps

The focus is on educational clarity - understanding how neuralization
enables explainability for traditional ML models.

Usage:
    python demo_mnist.py

Output:
    Saves figure to figures/mnist_explanations.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from sklearn.datasets import fetch_openml
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics.pairwise import euclidean_distances

from neuralised_svm import NeuralisedSVM, compute_heuristic_eta_svm
from neuralised_knn import NeuralisedKNN, compute_heuristic_eta_knn, compute_heuristic_kappa

# =============================================================================
# Configuration
# =============================================================================

RANDOM_SEED = 42
MAX_SAMPLES = 5000  # Maximum total samples to use
TEST_SIZE = 0.2     # Fraction for test set
N_EXPLAIN = 5       # Number of samples to explain

np.random.seed(RANDOM_SEED)

# =============================================================================
# Step 1: Load MNIST data (digits 4 and 9 only)
# =============================================================================

print("=" * 70)
print("Step 1: Loading MNIST data (digits 4 and 9)")
print("=" * 70)

# Load MNIST dataset
print("Fetching MNIST from OpenML (this may take a moment)...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
X_all, y_all = mnist.data, mnist.target.astype(int)

# Filter to only digits 4 and 9
mask = (y_all == 4) | (y_all == 9)
X_filtered = X_all[mask]
y_filtered = y_all[mask]

# Convert labels to {-1, +1}
# Digit 4 -> -1, Digit 9 -> +1
y_binary = np.where(y_filtered == 4, -1, 1)

print(f"Total samples with digits 4 and 9: {len(X_filtered)}")

# Subsample if needed
if len(X_filtered) > MAX_SAMPLES:
    indices = np.random.choice(len(X_filtered), MAX_SAMPLES, replace=False)
    X_filtered = X_filtered[indices]
    y_binary = y_binary[indices]
    print(f"Subsampled to: {MAX_SAMPLES} samples")

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X_filtered, y_binary, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y_binary
)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Feature dimensions: {X_train.shape[1]} (28x28 flattened)")
print(f"Class distribution (train): {np.sum(y_train == -1)} fours, {np.sum(y_train == 1)} nines")

# =============================================================================
# Step 2: Distance normalization preprocessing
# =============================================================================

print("\n" + "=" * 70)
print("Step 2: Applying distance normalization")
print("=" * 70)

def distance_normalization(X_train, X_test):
    """
    Normalize data so that median pairwise squared distance equals 1.

    This makes the gamma parameter more interpretable:
    - gamma=1 means the kernel "sees" the typical distance between points
    - gamma>1 means sharper kernel (more local)
    - gamma<1 means smoother kernel (more global)

    Args:
        X_train: Training data
        X_test: Test data

    Returns:
        X_train_normed, X_test_normed: Normalized data
    """
    # Compute pairwise squared distances on training data
    D_squared = euclidean_distances(X_train, X_train) ** 2

    # Find the median (excluding diagonal zeros)
    # We use all pairwise distances including diagonal for simplicity
    median_dist_sq = np.median(D_squared)

    # Scale factor to make median = 1
    scale = np.sqrt(median_dist_sq)

    # Normalize both train and test
    X_train_normed = X_train / scale
    X_test_normed = X_test / scale

    # Verify normalization worked
    D_check = euclidean_distances(X_train_normed, X_train_normed) ** 2
    new_median = np.median(D_check)

    print(f"Original median squared distance: {median_dist_sq:.4f}")
    print(f"After normalization: {new_median:.4f} (should be ~1.0)")

    return X_train_normed, X_test_normed

X_train_norm, X_test_norm = distance_normalization(X_train, X_test)

# =============================================================================
# Step 3: Train SVM and KNN with grid search
# =============================================================================

print("\n" + "=" * 70)
print("Step 3: Training models with hyperparameter tuning")
print("=" * 70)

# --- SVM Training ---
print("\n--- Training SVM ---")
svm_param_grid = {
    'gamma': [0.1, 0.3, 1.0, 3.0, 10.0],
    'C': [0.1, 1.0, 10.0]
}

svm_grid = GridSearchCV(
    SVC(kernel='rbf'),
    svm_param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=1,
    verbose=1
)
svm_grid.fit(X_train_norm, y_train)

best_svm = svm_grid.best_estimator_
print(f"\nBest SVM parameters: gamma={best_svm.gamma}, C={best_svm.C}")
print(f"SVM Training accuracy: {best_svm.score(X_train_norm, y_train):.4f}")
print(f"SVM Test accuracy: {best_svm.score(X_test_norm, y_test):.4f}")
print(f"Number of support vectors: {len(best_svm.support_vectors_)}")

# --- KNN Training ---
print("\n--- Training KNN ---")
knn_param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11]
}

knn_grid = GridSearchCV(
    KNeighborsClassifier(),
    knn_param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=1,
    verbose=1
)
knn_grid.fit(X_train_norm, y_train)

best_knn = knn_grid.best_estimator_
print(f"\nBest KNN parameters: k={best_knn.n_neighbors}")
print(f"KNN Training accuracy: {best_knn.score(X_train_norm, y_train):.4f}")
print(f"KNN Test accuracy: {best_knn.score(X_test_norm, y_test):.4f}")

# =============================================================================
# Step 4: Verify neuralization preserves classifications
# =============================================================================

print("\n" + "=" * 70)
print("Step 4: Verifying neuralization preserves classifications")
print("=" * 70)

# --- Neuralize SVM ---
print("\n--- Neuralizing SVM ---")
neural_svm = NeuralisedSVM(best_svm)

# Compare decision functions
svm_original = best_svm.decision_function(X_test_norm)
svm_neural = neural_svm.forward(X_test_norm, with_intercept=True)

# Check sign agreement
svm_sign_original = np.sign(svm_original)
svm_sign_neural = np.sign(svm_neural)

svm_agreement = np.mean(svm_sign_original == svm_sign_neural)

print(f"SVM sign agreement: {svm_agreement * 100:.2f}%")

# --- Neuralize KNN ---
print("\n--- Neuralizing KNN ---")
neural_knn = NeuralisedKNN(best_knn)

# For KNN, compare using a simple scoring function
def knn_score(knn, X):
    """Compute a soft score for KNN (mean label of neighbors)."""
    neighbor_indices = knn.kneighbors(X, return_distance=False)
    # Note: knn._y stores encoded indices (0, 1), not actual labels (-1, +1)
    # We need to map back to original labels using knn.classes_
    neighbor_label_indices = knn._y[neighbor_indices]
    neighbor_labels = knn.classes_[neighbor_label_indices]  # Convert to actual labels (-1, +1)
    return np.mean(neighbor_labels, axis=1)

knn_original = knn_score(best_knn, X_test_norm)
knn_neural = neural_knn.forward(X_test_norm, kappa=0)

# Check sign agreement
knn_sign_original = np.sign(knn_original)
knn_sign_neural = np.sign(knn_neural)

knn_agreement = np.mean(knn_sign_original == knn_sign_neural)

print(f"KNN sign agreement: {knn_agreement * 100:.2f}%")

# =============================================================================
# Step 5: Compute eta heuristics
# =============================================================================

print("\n" + "=" * 70)
print("Step 5: Computing explanation hyperparameters (eta)")
print("=" * 70)

# SVM eta depends on gamma
svm_gamma = best_svm.gamma
svm_eta = compute_heuristic_eta_svm(svm_gamma)

print("\n--- SVM Eta Heuristic ---")
print(f"SVM gamma: {svm_gamma}")
print(f"log10(gamma): {np.log10(svm_gamma):.2f}")
print(f"Heuristic eta: {svm_eta}")
print("\nEta interpretation:")
print("  eta=0: Pure Gradient x Input (GI) - good for smooth kernels")
print("  eta=1: Pure Midpoint rule - good for sharp kernels")
print(f"  eta={svm_eta}: Hybrid mixing both rules")

# KNN eta is fixed
knn_k = best_knn.n_neighbors
knn_eta = compute_heuristic_eta_knn(knn_k)
knn_kappa = compute_heuristic_kappa(knn_k)

print("\n--- KNN Eta Heuristic ---")
print(f"KNN k: {knn_k}")
print(f"Heuristic eta: {knn_eta} (fixed for KNN)")
print(f"Heuristic kappa: {knn_kappa} (neighbor range)")

# =============================================================================
# Step 6: Generate explanations for sample images
# =============================================================================

print("\n" + "=" * 70)
print("Step 6: Generating hybrid explanations")
print("=" * 70)

# Select samples: mix of 4s and 9s, correctly classified
correct_mask = best_svm.predict(X_test_norm) == y_test
fours_idx = np.where((y_test == -1) & correct_mask)[0]
nines_idx = np.where((y_test == 1) & correct_mask)[0]

# Take some of each
n_fours = min(N_EXPLAIN // 2, len(fours_idx))
n_nines = N_EXPLAIN - n_fours

selected_idx = np.concatenate([
    np.random.choice(fours_idx, n_fours, replace=False),
    np.random.choice(nines_idx, n_nines, replace=False)
])

X_explain = X_test_norm[selected_idx]
y_explain = y_test[selected_idx]
X_explain_original = X_test[selected_idx]  # For visualization (unnormalized)

print(f"Explaining {N_EXPLAIN} samples: {n_fours} fours and {n_nines} nines")

# Compute SVM explanations
print("\nComputing SVM hybrid explanations...")
svm_explanations = neural_svm.explain(
    X_explain,
    rule="hybrid",
    eta=svm_eta,
    beta=svm_gamma,  # Use gamma as beta (heuristic)
    with_intercept=True
)

# Compute KNN explanations
print("Computing KNN hybrid explanations...")
knn_explanations = neural_knn.explain(
    X_explain,
    rule="hybrid",
    eta=knn_eta
)

print("Explanations computed!")

# =============================================================================
# Step 7: Visualize and save heatmaps
# =============================================================================

print("\n" + "=" * 70)
print("Step 7: Creating visualization")
print("=" * 70)

fig, axes = plt.subplots(3, N_EXPLAIN, figsize=(3 * N_EXPLAIN, 9))

# Row labels on the left side
row_labels = [
    'Original\nImage',
    f'SVM Explanation\n(γ={svm_gamma}, η={svm_eta})',
    f'KNN Explanation\n(k={knn_k}, η={knn_eta})'
]

for i in range(N_EXPLAIN):
    # Get the original image (for display)
    img = X_explain_original[i].reshape(28, 28)
    label = "4" if y_explain[i] == -1 else "9"

    # Get explanations and reshape to image
    svm_exp = svm_explanations[i].reshape(28, 28)
    knn_exp = knn_explanations[i].reshape(28, 28)

    # Row 1: Original image
    ax1 = axes[0, i]
    ax1.imshow(img, cmap='gray')
    ax1.set_title(f'Digit: {label}', fontsize=12)
    ax1.axis('off')

    # Row 2: SVM explanation
    ax2 = axes[1, i]
    # Use diverging colormap centered at 0
    vmax = max(abs(svm_exp.min()), abs(svm_exp.max()))
    if vmax > 0:
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    else:
        norm = None
    im2 = ax2.imshow(svm_exp, cmap='seismic', norm=norm)
    ax2.axis('off')

    # Row 3: KNN explanation
    ax3 = axes[2, i]
    vmax = max(abs(knn_exp.min()), abs(knn_exp.max()))
    if vmax > 0:
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    else:
        norm = None
    im3 = ax3.imshow(knn_exp, cmap='seismic', norm=norm)
    ax3.axis('off')

# Add row labels on the left
for row_idx, label in enumerate(row_labels):
    axes[row_idx, 0].annotate(
        label, xy=(-0.1, 0.5), xycoords='axes fraction',
        fontsize=11, ha='right', va='center',
        rotation=0, fontweight='bold'
    )

# Add overall title - clarify that explanations are for the positive class (digit 9)
fig.suptitle(
    'Neuralized Explanations (Hybrid Rule)\n'
    'Explaining the positive class (digit 9)',
    fontsize=14, y=1.02
)

# Add colorbar explanation at the bottom
fig.text(0.5, -0.02,
         'Red = evidence for digit 9 (positive class) | Blue = evidence for digit 4 (negative class)',
         ha='center', fontsize=11)

plt.tight_layout()

# Save figure
output_path = 'figures/mnist_explanations.png'
os.makedirs('figures', exist_ok=True)
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Figure saved to: {output_path}")

plt.show()

# =============================================================================
# Summary
# =============================================================================

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"""
Dataset:
  - MNIST digits 4 vs 9
  - {len(X_train)} training samples, {len(X_test)} test samples
  - Distance normalization applied

SVM Model:
  - Best gamma: {svm_gamma}
  - Best C: {best_svm.C}
  - Test accuracy: {best_svm.score(X_test_norm, y_test):.4f}
  - Explanation eta: {svm_eta}

KNN Model:
  - Best k: {knn_k}
  - Test accuracy: {best_knn.score(X_test_norm, y_test):.4f}
  - Explanation eta: {knn_eta}
  - Explanation kappa: {knn_kappa}

Explanations:
  - Method: Hybrid rule (combining GI and Midpoint)
  - {N_EXPLAIN} sample images explained
  - Saved to: {output_path}
""")
