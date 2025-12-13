"""
Neuralised SVM - A Neural Network View of Support Vector Machines

This module converts a trained sklearn SVM into a "neuralized" representation
that enables the application of neural network explanation methods (like LRP).

The key idea: An RBF SVM's decision function can be rewritten as a difference
of two log-sum-exp terms, which resembles a neural network with softmax layers.

Original SVM decision function:
    f(x) = sum_i alpha_i * K(x, x_i) + b
         = sum_i alpha_i * exp(-gamma * ||x - x_i||^2) + b

Neuralized form:
    g(x) = logsumexp(-gamma * z_pos) - logsumexp(-gamma * z_neg)

where z_pos and z_neg are distance-based terms to positive and negative
support vectors respectively.

This reformulation allows us to:
1. Apply gradient-based explanation methods
2. Use LRP-style relevance propagation
3. Combine multiple explanation strategies (GI, midpoint, hybrid)

Author: Educational version adapted from research code
"""

import numpy as np
from scipy.special import logsumexp, softmax


class NeuralisedSVM:
    """
    A neuralized wrapper around a trained sklearn SVC.

    The neuralization converts the SVM into a form that resembles a neural network,
    enabling the use of explanation methods typically applied to neural networks.

    Attributes:
        gamma: RBF kernel parameter from the original SVM
        x_sup_pos: Support vectors with positive dual coefficients (class +1)
        x_sup_neg: Support vectors with negative dual coefficients (class -1)
        alphas_pos: Positive dual coefficients (magnitude)
        alphas_neg: Negative dual coefficients (magnitude, stored as positive)
        intercept_: Bias term from the original SVM
    """

    def __init__(self, svc):
        """
        Initialize the neuralized SVM from a trained sklearn SVC.

        Args:
            svc: A trained sklearn.svm.SVC object with RBF kernel
        """
        # Store reference to original model and extract key parameters
        self.original_svc = svc
        self.intercept_ = svc.intercept_[0] if len(svc.intercept_) > 0 else 0
        self.gamma = svc.gamma

        # Extract dual coefficients (alphas)
        # In sklearn, dual_coef_ has shape (1, n_support_vectors) for binary classification
        # Positive alphas correspond to positive class support vectors
        # Negative alphas correspond to negative class support vectors
        dual_coefs = svc.dual_coef_[0]

        self.alphas_pos = dual_coefs[dual_coefs > 0]
        self.alphas_neg = np.abs(dual_coefs[dual_coefs < 0])  # Store as positive values

        # Extract support vectors, separated by class
        self.x_sup_pos = svc.support_vectors_[dual_coefs > 0]
        self.x_sup_neg = svc.support_vectors_[dual_coefs < 0]

        self.num_pos = len(self.alphas_pos)
        self.num_neg = len(self.alphas_neg)

    def compute_z(self, x, with_intercept=False):
        """
        Compute the "z" values - distance-based terms to support vectors.

        The z values combine squared Euclidean distances with alpha weights:
            z_i = ||x - x_i||^2 - log(alpha_i) / gamma

        This formulation allows the SVM decision function to be written as:
            f(x) ~ exp(-gamma * z_pos) - exp(-gamma * z_neg)

        Args:
            x: Input samples, shape (n_samples, n_features)
            with_intercept: If True, incorporate the bias term into z

        Returns:
            z_pos: Distance terms to positive support vectors, shape (n_samples, n_pos_sv)
            z_neg: Distance terms to negative support vectors, shape (n_samples, n_neg_sv)
        """
        # Compute squared distances from each input to each positive support vector
        # diff shape: (n_samples, n_pos_sv, n_features)
        sv_pos_diff = x[:, None] - self.x_sup_pos[None]
        sv_pos_sq_distance = np.linalg.norm(sv_pos_diff, axis=2) ** 2

        # Combine distance with alpha weight
        # The log(alpha)/gamma term shifts the "effective distance"
        z_pos = sv_pos_sq_distance - np.log(self.alphas_pos)[None] / self.gamma

        # Same for negative support vectors
        sv_neg_diff = x[:, None] - self.x_sup_neg[None]
        sv_neg_sq_distance = np.linalg.norm(sv_neg_diff, axis=2) ** 2
        z_neg = sv_neg_sq_distance - np.log(self.alphas_neg)[None] / self.gamma

        # Optionally incorporate the intercept as an additional "virtual" support vector
        if with_intercept:
            if self.intercept_ >= 0:
                # Positive intercept acts like an additional positive support vector
                z_pos_bias = -np.log(self.intercept_ + 1e-10) / self.gamma
                z_pos = np.concatenate([z_pos, np.full((len(x), 1), z_pos_bias)], axis=1)
            else:
                # Negative intercept acts like an additional negative support vector
                z_neg_bias = -np.log(-self.intercept_ + 1e-10) / self.gamma
                z_neg = np.concatenate([z_neg, np.full((len(x), 1), z_neg_bias)], axis=1)

        return z_pos, z_neg

    def forward(self, x, with_intercept=False):
        """
        Compute the neuralized forward pass.

        This is the "neural network" view of the SVM decision:
            g(x) = logsumexp(-gamma * z_pos) - logsumexp(-gamma * z_neg)

        The sign of g(x) matches the original SVM classification, though
        the magnitude may differ.

        Args:
            x: Input samples, shape (n_samples, n_features)
            with_intercept: If True, incorporate bias into the computation

        Returns:
            g: Neuralized output values, shape (n_samples,)
        """
        z_pos, z_neg = self.compute_z(x, with_intercept=with_intercept)

        # Log-sum-exp is a smooth approximation to max
        # This gives a differentiable version of the SVM decision
        g = logsumexp(-self.gamma * z_pos, axis=1) - logsumexp(-self.gamma * z_neg, axis=1)

        return g

    def compute_point_pair_weights(self, x, beta=None, with_intercept=False):
        """
        Compute the "weights" over support vectors.

        These weights determine how much each support vector contributes
        to the explanation for a given input. They are computed as softmax
        over the negative z values (so closer/more relevant SVs get higher weight).

        The beta parameter controls the "temperature" of the softmax:
        - Higher beta = sharper focus on nearest support vectors
        - Lower beta = more uniform distribution over support vectors

        Args:
            x: Input samples, shape (n_samples, n_features)
            beta: Softmax temperature (default: same as gamma)
            with_intercept: If True, incorporate bias term

        Returns:
            p_pos: Weights over positive SVs, shape (n_samples, n_pos_sv)
            p_neg: Weights over negative SVs, shape (n_samples, n_neg_sv)
        """
        if beta is None:
            beta = self.gamma  # Default: use same sharpness as the kernel

        z_pos, z_neg = self.compute_z(x, with_intercept=with_intercept)

        # Softmax: points with smaller z (closer/more important) get higher weight
        # Each row sums to 1
        p_pos = softmax(-beta * z_pos, axis=1)
        p_neg = softmax(-beta * z_neg, axis=1)

        return p_pos, p_neg

    def explain(self, x, rule="hybrid", eta=0.5, beta=None, with_intercept=False):
        """
        Compute feature-wise explanations using neuralized LRP.

        Three explanation rules are available:

        1. GI (Gradient x Input):
           - Based on the gradient of the neuralized function
           - R = x * 2 * (weighted_mean_pos_sv - weighted_mean_neg_sv)
           - Works well when model is near-linear
           - Implicit reference point is data origin

        2. Midpoint:
           - Implemented as squared differences to support vectors
           - R = weighted_sq_diff_neg - weighted_sq_diff_pos
           - Reference point to each detection unit is midpoint spanned between pair of support vectors
           - Best suited when model is highly non-linear;
                i.e. detection units in different positions in input space require different reference poitns

        3. Hybrid:
           - Combines GI and midpoint: R = (1-eta)*GI + eta*midpoint
           - eta controls the balance (0=pure GI, 1=pure midpoint)

        For the derivation of the rule implementations see Supplementary Note E: Deriving efficient LRP explanations

        Interpretation of explanations:
        - Positive values: feature contributes toward positive class
        - Negative values: feature contributes toward negative class

        Args:
            x: Input samples, shape (n_samples, n_features)
            rule: One of "GI", "midpoint", or "hybrid"
            eta: Mixing parameter for hybrid rule (0 to 1)
            beta: Softmax temperature for weights (default: gamma)
            with_intercept: Whether to include bias in explanation

        Returns:
            R: Relevance scores, shape (n_samples, n_features)
        """
        # Compute weights over support vectors
        p_pos, p_neg = self.compute_point_pair_weights(x, beta=beta, with_intercept=with_intercept)

        if rule == "GI":
            # Gradient x Input rule
            # The weighted mean of support vector positions determines the "gradient direction"

            if not with_intercept:
                # Simple case: just weight the actual support vectors
                weighted_mean_sv_pos = np.einsum("ni, id -> nd", p_pos, self.x_sup_pos)
                weighted_mean_sv_neg = np.einsum("nj, jd -> nd", p_neg, self.x_sup_neg)
            else:
                # With intercept: the bias acts as a "virtual support vector" at position x
                # We need to include this virtual SV in the weighted mean

                # Expand support vectors to (n_samples, n_sv, n_features)
                x_sup_pos_expanded = np.repeat(self.x_sup_pos[None], len(x), axis=0)
                x_sup_neg_expanded = np.repeat(self.x_sup_neg[None], len(x), axis=0)

                if self.intercept_ >= 0:
                    # Positive intercept: add virtual positive SV at position x
                    # x[:, None] has shape (n_samples, 1, n_features)
                    x_sup_pos_expanded = np.concatenate([x_sup_pos_expanded, x[:, None]], axis=1)
                else:
                    # Negative intercept: add virtual negative SV at position x
                    x_sup_neg_expanded = np.concatenate([x_sup_neg_expanded, x[:, None]], axis=1)

                # Compute weighted mean including the virtual SV
                # p_pos/p_neg already include the bias weight from compute_point_pair_weights
                weighted_mean_sv_pos = (p_pos[:, :, None] * x_sup_pos_expanded).sum(axis=1)
                weighted_mean_sv_neg = (p_neg[:, :, None] * x_sup_neg_expanded).sum(axis=1)

            # The "gradient" direction is the difference of weighted means
            gradient_direction = 2 * (weighted_mean_sv_pos - weighted_mean_sv_neg)

            # Multiply by input (Gradient x Input)
            R = x * gradient_direction

        elif rule == "midpoint":
            # Midpoint rule - uses squared differences to support vectors
            # Essentially, each detection unit uses a different reference point - the midpoint between the
            # pair of spanning support vectors.
            # This makes sense if the decision boundary changes (non-linear model) and different points
            # in input space need different "perspectives", references

            # Compute squared difference from x to each support vector, per feature
            # Shape: (n_samples, n_sv, n_features)
            squared_diff_pos = (x[:, None] - self.x_sup_pos[None]) ** 2
            squared_diff_neg = (x[:, None] - self.x_sup_neg[None]) ** 2

            if not with_intercept:
                # Simple case: weight the squared differences
                weighted_sq_diff_pos = np.einsum("ni, nid -> nd", p_pos, squared_diff_pos)
                weighted_sq_diff_neg = np.einsum("nj, njd -> nd", p_neg, squared_diff_neg)
            else:
                # With intercept: add virtual SV at position x
                # The squared diff from x to x is zero, so we append zeros
                zeros = np.zeros((len(x), 1, x.shape[1]))

                if self.intercept_ >= 0:
                    squared_diff_pos = np.concatenate([squared_diff_pos, zeros], axis=1)
                else:
                    squared_diff_neg = np.concatenate([squared_diff_neg, zeros], axis=1)

                weighted_sq_diff_pos = np.einsum("ni, nid -> nd", p_pos, squared_diff_pos)
                weighted_sq_diff_neg = np.einsum("nj, njd -> nd", p_neg, squared_diff_neg)

            # Relevance: features where x is far from neg SVs but close to pos SVs
            # contribute positively to the positive class
            R = weighted_sq_diff_neg - weighted_sq_diff_pos

        elif rule == "hybrid":
            # Hybrid rule: weighted combination of GI and midpoint
            R_GI = self.explain(x, rule="GI", beta=beta, with_intercept=with_intercept)
            R_midpoint = self.explain(x, rule="midpoint", beta=beta, with_intercept=with_intercept)

            R = (1 - eta) * R_GI + eta * R_midpoint

        else:
            raise ValueError(f"Unknown rule: {rule}. Use 'GI', 'midpoint', or 'hybrid'.")

        return R


# =============================================================================
# Heuristic functions for hyperparameter selection
# =============================================================================

def compute_heuristic_eta_svm(gamma):
    """
    Compute the recommended eta value based on SVM gamma.

    The heuristic is based on empirical observations:
    - Small gamma (smooth kernel): GI works well -> eta near 0
    - Large gamma (sharp kernel): midpoint more stable -> eta near 1

    The transition happens around gamma = 1 (after distance normalization).

    Args:
        gamma: The RBF kernel parameter of the SVM

    Returns:
        eta: Recommended mixing parameter (0 to 1)
    """
    log_gamma = np.log10(gamma)

    if log_gamma < -0.5:
        # Small gamma: pure GI
        return 0.0
    elif log_gamma > 1.5:
        # Large gamma: pure midpoint
        return 1.0
    else:
        # Linear interpolation in between
        return round(log_gamma * 0.4 + 0.4, 2)
