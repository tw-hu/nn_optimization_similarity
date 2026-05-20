"""
Script used to compute the similarity between activation vectors of various neural networks
"""

from pathlib import Path

import torch

INSTANCES_PER_CONFIG = 10 # the number of neural networks trained on each hyperparameter configuration

for (lr_a, lr_b) in list(itertools.product()):
    for (s_a, s_b) in list(itertools.product(range(10), repeat=2)):
        file_a = f"trainmode_run_lr{lr1}_seed{s_a}"
        file_b = f"trainmode_run_lr{lr2}_seed{s_b}"

    path_a = Path(f"experiments/{file_a}/activations/epoch100.pt")
    actv_a = torch.load(path_a) # {layer_name: [n_samples, n_channels, dim_x, dim_y]}

    compute_similarity(activations1, activations2)