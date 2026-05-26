"""
Scripts used for analysis of neural network activation matrices
"""
import matplotlib.colors as mcolors
import numpy as np
from sklearn.manifold import MDS, Isomap

from typing import Dict

from payload.models.ConvClassifier import LAYERS

COLORS = ['#B38B00', '#FF6666', '#800000', '#341F16', '#BA55D3']
cmap_name = 'yellow-to-maroon'
custom_cmap = mcolors.LinearSegmentedColormap.from_list(cmap_name, COLORS)

"""
File structures:

similarity.pt: {(La_net_1, Lb_net 2): sim_val}
state_dict.pt: {
                "metrics": {"train/loss": loss, "val/loss": loss}
               }
"""

### Similarity ###

def get_agg_metric(s: str) -> callable:
    if s.upper() == "MDS":
        return compute_mean_diag_similarity
    elif s.upper() == "BDS":
        return compute_banded_diag_similarity
    elif s.upper() == "TTS":
        return compute_trace_to_sum
    else:
        raise ValueError("Aggregate metric must be one of 'MDS', 'BDS', and 'TTS.'")

def compute_mean_diag_similarity(M: np.ndarray | list) -> float:
    M = np.array(M)
    L = len(M)
    return np.trace(M)/L

def compute_banded_diag_similarity(M: np.ndarray | list, band: int = 1) -> float:
    M = np.array(M)
    L = len(M)
    band_vals = []
    for i in range(-band, band+1):
        diag = np.diag(M, k=i)
        if len(diag) > 0:
            band_vals.append(np.max(diag))
    return float(np.mean(band_vals)) if band_vals else 0.0

def compute_trace_to_sum(M: np.ndarray | list) -> float:
    M = np.array(M)
    return np.trace(M)/np.sum(M)

def to_distance(sim: float) -> float:
    assert 0 <= sim <= 1, "similarity value must be in [0, 1]"
    return 1 - sim

### Visualization ### 

def distances_to_coords(d: Dict, method: str = "MDS", k_nn: int = 5) -> Dict:
    """
    Input:
        Dict: {((label_1, seed_1), (label_2, seed_2)): distance}
    Output:
        Dict: {(seed, label): (x, y)}
    """

    # Extract all unique (label, seed) items
    nodes = set()
    for (u, v) in d.keys():
        nodes.add(u)
        nodes.add(v)
    nodes_list = list(nodes)
    n = len(nodes_list)

    node_to_idx = {node: i for i, node in enumerate(nodes_list)}

    if method.lower() == "mds":
        # Construct distance matrix
        dist_matrix = np.zeros((n, n))
        for (u, v), distance in d.items():
            i = node_to_idx[u]
            j = node_to_idx[v]
            dist_matrix[i, j] = distance
            dist_matrix[j, i] = distance

        mds = MDS(
            n_components=2, 
            metric='precomputed', 
            random_state=0,
            normalized_stress='auto',
            init='classical_mds',
            n_init=1
        )
        embedding = mds.fit_transform(dist_matrix)

    elif method.lower() == "isomap":
        dist_matrix = np.full((n, n), np.inf)
        np.fill_diagonal(dist_matrix, 0.0)
        for (u, v), distance in d.items():
            i = node_to_idx[u]
            j = node_to_idx[v]
            dist_matrix[i, j] = distance
            dist_matrix[j, i] = distance
        
        actual_n_neighbors = min(k_nn, n - 1) # Ensure k is strictly less than the number of nodes
        iso = Isomap(
            n_neighbors=actual_n_neighbors,
            n_components=2, 
            metric='precomputed'
        )
        embedding = iso.fit_transform(dist_matrix)

    # Format the output to {(seed, label): (x, y)}
    coords = {}
    for i, node in enumerate(nodes_list): 
        label, seed = node
        coords[(label, seed)] = (float(embedding[i, 0]), float(embedding[i, 1]))
        
    return coords

### Utils ###

def pt_to_matrix(pt: Dict) -> np.ndarray:
    M = np.zeros((len(LAYERS), len(LAYERS)))
    for i, La in enumerate(LAYERS):
        for j, Lb in enumerate(LAYERS):
            M[i, j] = pt[La, Lb]
    return M

