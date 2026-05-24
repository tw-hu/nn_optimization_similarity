"""
Scripts for computing similarity
"""
from collections import OrderedDict
import itertools
from pathlib import Path

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm

import torch
from torch import Tensor

def epochs_to_states(epochs: int, partitions: int) -> list:
    if epochs % partitions != 0:
        raise ValueError("Incompatible values of 'epochs' and 'partitions.'")
    intv = int(epochs/partitions)
    return [f"epoch_{intv * n}" for n in range(partitions)]

class SimilarityComputer():
    """
    A class which uses neuron activations to compute and visualize layer similarities
    All similarity measures are implemented on the conventions of Kornblith et al. (2019)
    """
    def __init__(
            self,
            activations_a: OrderedDict,
            activations_b: OrderedDict,
            experiment_name: str = None,
            output_dir: str = "outputs",
            model_state: str = "final"
            ):
        self.activations_a, self.activations_b = activations_a, activations_b # {layer_name: [n_samples, c, h, w]}
        self.layers_a, self.layers_b = list(activations_a), list(activations_b)
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_state = model_state
        self.similarities = {}

        self._metric: str = None

    def compute_similarities(self, metric: str = "cka"):
        if metric == "cka":
            _compute_similarity = self._compute_cka
        elif metric == "pwcca":
            _compute_similarity = self._compute_pwcca
        else:
            raise ValueError("Not a valid metric. Use either 'cka' or 'pwcca.'")
        
        self._metric = metric
        for La, Lb in itertools.product(list(self.activations_a.keys()), self.activations_b.keys()):
            self.similarities[(La, Lb)] = _compute_similarity(self.activations_a[La], self.activations_b[Lb])

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
    
    def svd_prune(self, K: torch.Tensor, threshold: float = 0.99) -> torch.Tensor:
        """
        Prunes the matrix K to the subspace which explains a threshold amount of variance of K
        """
        K_centered = K - K.mean(dim=0)
        U, S, _ = torch.linalg.svd(K_centered, full_matrices=False)
        eig_squared = S ** 2
        explained_var = eig_squared / torch.sum(eig_squared)
        cum_var = torch.cumsum(explained_var, dim=0)
    
        k = (cum_var >= threshold).nonzero(as_tuple=True)[0][0].item() + 1
        K_pruned = U[:, :k] * S[:k]
        return K_pruned
    
    def _compute_pwcca(self, features_a: Tensor, features_b: Tensor, keep_variance: float = 0.99, epsilon: float = 1e-8) -> float:
        """
        Computes Projected Weighted Canonical Correlation Analysis (PWCCA) metric for two
        activation matrices with shape (n_samples, c, h, w)
        """
        if features_a.dim() == 4:
            features_a = features_a.view(features_a.size(0), -1) # flatten activation into (n_samples, D)
        if features_b.dim() == 4:
            features_b = features_b.view(features_b.size(0), -1)

        pruned_a = self._svd_prune(features_a, threshold=keep_variance)
        pruned_b = self._svd_prune(features_b, threshold=keep_variance)

        n, da = pruned_a.shape
        _, db = pruned_b.shape

        centered_features_a = pruned_a - pruned_a.mean(dim=0)
        centered_features_b = pruned_b - pruned_b.mean(dim=0)

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
    
    def write_similarities(self, output_folder: Path | str = None) -> None:
        """
        Writes similarity dictionary to output folder for easy downstream retrieval and analysis
        """
        output_dir = Path(output_folder if output_folder is not None else self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_dir = output_dir / f"{self.experiment_name}_{self._metric}_{self.model_state}.pt"
        torch.save(self.similarities, output_dir)

    def plot_similarities(self, title: str = "Title", output_folder: str = None) -> None:
        """
        Generates a plot of similarities, with activations_a on the x-axis and activations_b on the y-axis.
        """
        if not self._metric:
            raise ValueError("Compute similarity with metric pwcca/cka first.")

        if self._metric == "cka":
            metric_str = "CKA"
        elif self._metric == "pwcca":
            metric_str = "PWCCA"

        vals = np.zeros((len(self.layers_b), len(self.layers_a)))
        for i, La in enumerate(self.layers_a):
            for j, Lb in enumerate(self.layers_b):
                vals[i, j] = self.similarities[La, Lb]

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(vals, cmap='viridis')

        ax.set_xticks(np.arange(len(self.layers_b)), labels=self.layers_b)
        ax.set_yticks(np.arange(len(self.layers_a)), labels=self.layers_a)
        ax.invert_yaxis()
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel(f"{metric_str} Similarity", rotation=-90, va="bottom")
        ax.set_title(title)
        plt.tight_layout()

        output_dir = Path(output_folder if output_folder is not None else self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_dir = output_dir / f"{self.experiment_name}_{self._metric}_{self.model_state}.png"
        plt.savefig(output_dir)