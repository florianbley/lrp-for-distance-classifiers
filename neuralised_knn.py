"""
Neuralised KNN - A Neural Network View of K-Nearest Neighbors

This module converts a trained sklearn KNN classifier into a "neuralized"
representation that enables gradient-based explanations.

The key idea: While KNN is not differentiable (it uses hard neighbor selection),
we can create a differentiable approximation by treating the k-nearest neighbors
of each class as "support vectors" and applying similar techniques as for SVMs.

For KNN, instead of softmax-weighted support vectors, we use uniform weights
over the selected neighbors, which preserves the local nature of KNN.

The neuralization allows us to:
1. Apply the same explanation methods as for SVMs
2. Use the hybrid (GI + midpoint) explanation rule
3. Compare SVM and KNN explanations on the same framework

Author: Educational version adapted from research code
"""

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

# Import the base class - we inherit the explain() method from NeuralisedSVM
from neuralised_svm import NeuralisedSVM


class NeuralisedKNN(NeuralisedSVM):
    """
    A neuralized wrapper around a trained sklearn KNeighborsClassifier.

    The neuralization treats the k-nearest neighbors of each class as
    "support vectors" and applies similar explanation techniques as for SVMs.

    Key difference from SVM:
    - Instead of softmax weights, uses uniform weights over selected neighbors
    - The "deciding" neighbors are those around the (k+1)/2-th nearest per class

    Attributes:
        k: Number of neighbors from the original KNN
        X_train: Training data from the KNN
        y_train: Training labels from the KNN
        x_sup_pos: Training points of the positive class
        x_sup_neg: Training points of the negative class
    """

    def __init__(self, knn):
        """
        Initialize the neuralized KNN from a trained sklearn KNeighborsClassifier.

        Args:
            knn: A trained sklearn.neighbors.KNeighborsClassifier object
        """
        # Store the original model
        self.origmodel = knn

        # Extract training data (sklearn stores these as private attributes)
        self.X_train = knn._fit_X
        self.y_train = knn._y
        self.k = knn.n_neighbors

        # For KNN, we don't have a meaningful gamma or intercept
        # Set dummy values for compatibility with parent class methods
        self.gamma = 1e-5  # Very small - not really used
        self.intercept_ = 0

        # Determine class labels
        unique_targets = np.unique(self.y_train)
        self.pos_class = np.max(unique_targets)  # Higher label = positive class
        self.neg_class = np.min(unique_targets)  # Lower label = negative class

        # Separate training points by class (these act as "support vectors")
        self.x_sup_pos = self.X_train[self.y_train.ravel() == self.pos_class]
        self.x_sup_neg = self.X_train[self.y_train.ravel() == self.neg_class]

        self.num_pos = len(self.x_sup_pos)
        self.num_neg = len(self.x_sup_neg)

    def compute_z(self, x):
        """
        Compute squared distances from inputs to all training points.

        For KNN, we don't need the alpha-weighted adjustment that SVM uses.
        We just compute pure squared Euclidean distances.

        Args:
            x: Input samples, shape (n_samples, n_features)

        Returns:
            z_pos: Squared distances to positive class points, shape (n_samples, n_pos)
            z_neg: Squared distances to negative class points, shape (n_samples, n_neg)
        """
        # Squared distances to positive class training points
        sv_pos_diff = x[:, None] - self.x_sup_pos[None]
        z_pos = np.linalg.norm(sv_pos_diff, axis=2) ** 2

        # Squared distances to negative class training points
        sv_neg_diff = x[:, None] - self.x_sup_neg[None]
        z_neg = np.linalg.norm(sv_neg_diff, axis=2) ** 2

        return z_pos, z_neg

    def forward(self, x, kappa=None):
        """
        Compute the neuralized KNN forward pass.

        Unlike SVM which uses log-sum-exp, KNN uses a weighted sum of distances
        to selected neighbors. The output represents how much "closer" the input
        is to one class versus the other.

        Args:
            x: Input samples, shape (n_samples, n_features)
            kappa: Number of neighbors around the deciding point to include
                   (default: (k-1)//2, which includes all deciding neighbors)

        Returns:
            g: Neuralized output, shape (n_samples,)
                Positive = closer to negative class (counterintuitive but matches SVM convention)
        """
        if kappa is None:
            kappa = (self.k - 1) // 2

        z_pos, z_neg = self.compute_z(x)
        p_pos, p_neg = self.compute_point_pair_weights(x, kappa=kappa)

        # Weighted sum of distances: g > 0 if closer to pos class than neg class
        # Note: convention matches SVM where we compare neg - pos
        g = (z_neg * p_neg).sum(axis=1) - (z_pos * p_pos).sum(axis=1)

        return g

    def compute_point_pair_weights(self, x, kappa=None, with_intercept=False, beta=None):
        """
        Compute uniform weights over selected neighbors.

        For KNN, we select neighbors around the "deciding" position - the
        (k+1)/2-th nearest neighbor of each class. This is where the
        classification decision is made.

        The kappa parameter controls how many neighbors to include:
        - kappa=0: Only the deciding neighbor (1 per class)
        - kappa=1: Deciding neighbor +/- 1 (3 per class)
        - kappa=(k-1)//2: All k neighbors (maximum)

        Args:
            x: Input samples, shape (n_samples, n_features)
            kappa: Radius around deciding neighbor to include
            with_intercept: Unused, for API compatibility
            beta: Unused, for API compatibility

        Returns:
            p_pos: Weights over positive class points, shape (n_samples, n_pos)
            p_neg: Weights over negative class points, shape (n_samples, n_neg)
        """
        if kappa is None:
            kappa = (self.k - 1) // 2

        # The "deciding" neighbor position: the (k+1)/2-th nearest
        # For k=5: deciding position is 3rd nearest (index 2, which is (5+1)//2 - 1 = 2)
        k_deciding = (self.k + 1) // 2

        # Compute squared distances to all training points
        D_pos = euclidean_distances(x, self.x_sup_pos) ** 2
        D_neg = euclidean_distances(x, self.x_sup_neg) ** 2

        # Initialize weight matrices
        p_pos = np.zeros((len(x), len(self.x_sup_pos)))
        p_neg = np.zeros((len(x), len(self.x_sup_neg)))

        # Number of neighbors we'll select (2*kappa + 1 in "donut" mode)
        n_neighbors = 2 * kappa + 1

        # Find indices of neighbors around the deciding position
        # "Donut" mode: select neighbors from position (k_deciding - kappa) to (k_deciding + kappa)
        sorted_pos_idx = np.argsort(D_pos, axis=1)
        sorted_neg_idx = np.argsort(D_neg, axis=1)

        # Select the donut of neighbors
        start_idx = k_deciding - 1 - kappa
        end_idx = k_deciding - 1 + kappa + 1
        selected_pos = sorted_pos_idx[:, start_idx:end_idx]
        selected_neg = sorted_neg_idx[:, start_idx:end_idx]

        # Assign uniform weights to selected neighbors
        np.put_along_axis(p_pos, selected_pos, 1.0 / n_neighbors, axis=1)
        np.put_along_axis(p_neg, selected_neg, 1.0 / n_neighbors, axis=1)

        # Verify weights sum to 1
        assert np.allclose(p_pos.sum(axis=1), 1.0), "Positive weights don't sum to 1"
        assert np.allclose(p_neg.sum(axis=1), 1.0), "Negative weights don't sum to 1"

        return p_pos, p_neg


# =============================================================================
# Heuristic functions for hyperparameter selection
# =============================================================================

def compute_heuristic_eta_knn(k):
    """
    Compute the recommended eta value for KNN.

    Unlike SVM where eta depends on gamma, for KNN the optimal eta
    has been found empirically to be around 0.8 regardless of k.

    This means KNN explanations benefit from a heavier weight on the
    midpoint rule compared to SVM.

    Args:
        k: Number of neighbors (unused, but kept for API consistency)

    Returns:
        eta: Recommended mixing parameter (fixed at 0.8)
    """
    return 0.8  # Fixed heuristic


def compute_heuristic_kappa(k):
    """
    Compute the recommended kappa value for KNN explanations.

    Kappa controls how many neighbors contribute to the explanation.
    The heuristic is to use all deciding neighbors: kappa = (k-1)//2

    For k=5: kappa=2, meaning neighbors 1,2,3 (the deciding region)
    For k=7: kappa=3, meaning neighbors 1,2,3,4 (the deciding region)

    Args:
        k: Number of neighbors from the KNN

    Returns:
        kappa: Recommended neighbor range parameter
    """
    return (k - 1) // 2
