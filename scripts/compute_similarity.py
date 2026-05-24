"""
Script used to compute the similarity between activation vectors of various neural networks
"""
import logging
from pathlib import Path

import torch
import hydra

from omegaconf import OmegaConf, DictConfig

from payload.analysis.SimilarityComputer import SimilarityComputer

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    opt_a, opt_b = cfg.net_1.optimizer, cfg.net_2.optimizer
    s_a, s_b = cfg.net_1.seed, cfg.net_2.seed

    if cfg.state == "final":
        model_states = ["final"]
    elif cfg.state == "epoch":
        opt_path = Path(f"configs/optimizer/{cfg.net_1.optimizer}.yaml")
        epochs = OmegaConf.load(opt_path).epochs if cfg.mode == "train" else 2
        model_states = [f"epoch_{n}" for n in range(0, epochs, 5)]

    for model_state in model_states:
        file_a = f"{cfg.mode.mode}mode_run_optimizer_{opt_a}_seed{s_a}"
        file_b = f"{cfg.mode.mode}mode_run_optimizer_{opt_b}_seed{s_b}"
        path_a = Path(f"experiments/{file_a}/activations/{model_state}.pt")
        path_b = Path(f"experiments/{file_b}/activations/{model_state}.pt")
        actv_a = torch.load(path_a, map_location=device) # {layer_name: [n_samples, n_channels, dim_x, dim_y]}
        actv_b = torch.load(path_b, map_location=device)

        name = f"{cfg.mode.mode}mode_opA_{opt_a}_{s_a}_opB_{opt_b}_{s_b}"

        similarity = SimilarityComputer(actv_a, actv_b, experiment_name=name, model_state=model_state)
        similarity.compute_similarities(metric=cfg.sim_metric)
        similarity.plot_similarities(title=f"Layer similarity at model state {model_state}")
        similarity.write_similarities()

if __name__ == "__main__":
    main()