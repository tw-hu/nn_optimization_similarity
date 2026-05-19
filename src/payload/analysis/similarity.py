"""
Scripts for computing similarity
"""

import torch
from torch import Tensor

def hsic(K: Tensor, L: Tensor):
    """Computes Hilbert-Schmidt Independence Criterion (HSIC)"""

    n = K.shape[0]
    H = torch.eye(n, device=K.device) - (1.0 / n) * torch.ones((n, n), device=K.device)
    K_centered = torch.matmul(torch.matmul(H, K), H)
    L_centered = torch.matmul(torch.matmul(H, L), H)

    hsic_val = torch.trace(torch.matmul(K_centered, L_centered)) / ((n - 1)**2)
    return hsic_val

def compute_cka(features_a: Tensor, features_b: Tensor) -> float:
    """
    Computes Centered Kernel Alignment (CKA) for two activation matrices.
    
    Args:
        features_a: Activations from Model A, shape (N, C, H, W) or (N, D).
        features_b: Activations from Model B, shape (N, C, H, W) or (N, D).
    """
    if features_a.dim() == 4:
        features_a = features_a.view(features_a.size(0), -1)
    if features_b.dim() == 4:
        features_b = features_b.view(features_b.size(0), -1)

    K = torch.matmul(features_a, features_a.t())
    L = torch.matmul(features_b, features_b.t())
    hsic_kl = hsic(K, L)
    hsic_kk = hsic(K, K)
    hsic_ll = hsic(L, L)
    cka_score = hsic_kl / torch.sqrt(hsic_kk * hsic_ll)
    
    return cka_score.item()