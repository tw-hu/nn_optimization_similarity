"""
Scripts for computing similarity
"""
from collections import OrderedDict
import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm

import torch
from torch import Tensor

class SimilarityComputer():
    """
    A class which uses neuron activations to compute and visualize layer similarities
    All similarity measures are implemented on the conventions of Kornblith et al. (2019)
    """
    def __init__(self, activations_a: OrderedDict, activations_b: OrderedDict, experiment_name: str = None, output_dir: str = "outputs"):
        self.activations_a, self.activations_b = activations_a, activations_b # {layer_name: [n_samples, c, h, w]}
        self.layers_a, self.layers_b = list(activations_a), list(activations_b)
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self._compute_similarity = None
        self.similarities = {}

    def compute_similarities(self, metric: str = "cka"):
        if metric == "cka":
            _compute_similarity = self._compute_cka
        if metric == "pwcca":
            _compute_similarity = self._compute_pwcca

        for La, Lb in itertools.product(list(self.activations_a.keys()), self.activations_b.keys()):
            self.similarities[(La, Lb)] = _compute_similarity(self.activations_a[La], self.activations_b[Lb])
        if self.output_dir:
            torch.save(self.similarities, self.output_dir / f"{metric}_similarity_dict.pt")

    def plot_similarities(self) -> None:
        """
        Generates a plot of similarities, with activations_a on the x-axis and activations_b on the y-axis.
        """
        x_vars, y_vars, vals = [], [], []
        for (x, y), val in self.similarities.items():
            x_vars.append(x)
            y_vars.append(y)
            vals.append(val)

        cmap = cm.get_cmap("viridis")
        norm = mcolors.Normalize(vmin=min(vals), vmax=max(vals))
        
        fig, ax = plt.subplots(figsize=(7, 4))
        scatter = ax.scatter(x_vars, y_vars, c=vals, cmap=cmap, norm=norm, s=500, marker="s")
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label("Similarities")
        if self.output_dir:
            plt.savefig(self.output_dir / f"{self.experiment_name}_similarities.png")
        plt.show()

    def _hsic(self, K: Tensor, L: Tensor):
        """Computes Hilbert-Schmidt Independence Criterion (HSIC)"""

        n = K.shape[0]
        H = torch.eye(n, device=K.device) - (1.0 / n) * torch.ones((n, n), device=K.device) # centering matrix
        K_centered = torch.matmul(torch.matmul(H, K), H)
        L_centered = torch.matmul(torch.matmul(H, L), H)

        hsic_val = torch.trace(torch.matmul(K_centered, L_centered)) / ((n - 1)**2)
        return hsic_val

    def _compute_cka(self, features_a: Tensor, features_b: Tensor) -> float:
        """
        Computes Centered Kernel Alignment (CKA) for two activation matrices with shape
        (n_samples, c, h, w)
        """
        if features_a.dim() == 4:
            features_a = features_a.view(features_a.size(0), -1) # flatten activation into (n_samples, D)
        if features_b.dim() == 4:
            features_b = features_b.view(features_b.size(0), -1)

        K = torch.matmul(features_a, features_a.t())
        L = torch.matmul(features_b, features_b.t())
        hsic_kl = self._hsic(K, L)
        hsic_kk = self._hsic(K, K)
        hsic_ll = self._hsic(L, L)
        cka_score = hsic_kl / torch.sqrt(hsic_kk * hsic_ll)
        
        return cka_score.item()
    
    def _compute_pwcca(features_a: Tensor, features_b: Tensor, epsilon: float = 1e-8) -> float:
        """
        Computes Projected Weighted Canonical Correlation Analysis (PWCCA) metric for two
        activation matrices with shape (n_samples, c, h, w)
        """
        if features_a.dim() == 4:
            features_a = features_a.view(features_a.size(0), -1) # flatten activation into (n_samples, D)
        if features_b.dim() == 4:
            features_b = features_b.view(features_b.size(0), -1)

        n, da = features_a.shape
        _, db = features_b.shape

        centered_features_a = features_a - features_a.mean(dim=0)
        centered_features_b = features_b - features_b.mean(dim=0)

        S_aa = (centered_features_a.T @ centered_features_a) / (n - 1) # (da, da)
        S_bb = (centered_features_b.T @ centered_features_b) / (n - 1)
        S_ab = (centered_features_a.T @ centered_features_b) / (n - 1)
        S_aa = S_aa + epsilon * torch.eye(da, device=features_a.device, dtype=features_a.dtype)
        S_bb = S_bb + epsilon * torch.eye(db, device=features_b.device, dtype=features_b.dtype)

        def _inv_sqrt(K: Tensor) -> Tensor:
            """Computes inverse square root K^{-1/2} of K"""
            evals, evecs = torch.linalg.eigh(K)
            evals = torch.clamp(evals, min=epsilon) 
            inv_sqrt = torch.diag(1.0 / torch.sqrt(evals))
            return evecs @ inv_sqrt @ evecs.T
        
        S_aa_inv, S_bb_inv = _inv_sqrt(S_aa), _inv_sqrt(S_bb)
        U, rho, Vh = torch.linalg.svd(S_aa_inv @ S_ab @ S_bb_inv, full_matrices=False)
        Z_a = centered_features_a @ S_aa_inv @ U # (n, da) @ (da, da) @ (da, k=min(da, db))
        alpha_a = centered_features_a.T @ Z_a
        weights = torch.sum(torch.abs(alpha_a), dim=0)
        weights = weights / torch.sum(weights)
        pwcca_score = torch.dot(weights, rho)

        return pwcca_score.item()