"""
Script used to compute the similarity between activation vectors of various neural networks
"""
from pathlib import Path
import itertools

import torch

from payload.analysis.SimilarityComputer import SimilarityComputer

INSTANCES_PER_CONFIG = 10 # the number of neural networks trained on each hyperparameter configuration
LRS = [1.25e-4, 1e-3, 8e-3]

lr_a, lr_b = 1e-3, 1e-3
s_a, s_b = 0, 0
mode = "dev"

file_a = f"{mode}mode_run_lr{lr_a}_seed{s_a}"
file_b = f"{mode}mode_run_lr{lr_b}_seed{s_b}"

path_a = Path(f"experiments/{file_a}/activations/final.pt")
actv_a = torch.load(path_a) # {layer_name: [n_samples, n_channels, dim_x, dim_y]}

path_b = Path(f"experiments/{file_b}/activations/final.pt")
actv_b = torch.load(path_b)

name = f"{mode}mode_run_lra{lr_a}_lrb{lr_b}_seeda{s_a}_seedb{s_b}"

similarity = SimilarityComputer(actv_a, actv_b, experiment_name=name)
similarity.compute_similarities("cka")
similarity.plot_similarities()
similarity.compute_similarities("pwcca")
similarity.plot_similarities()
