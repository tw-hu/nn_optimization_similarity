"""
Script used to compute the similarity between activation vectors of various neural networks
"""
import logging
from pathlib import Path

import torch
import hydra

from omegaconf import OmegaConf, DictConfig

from payload.analysis.SimilarityComputer import SimilarityComputer, epochs_to_states

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    opt_a, opt_b = cfg.net_1.optimizer, cfg.net_2.optimizer
    s_a, s_b = cfg.net_1.seed, cfg.net_2.seed
    name = f"{cfg.mode.mode}mode_opA_{opt_a}_{s_a}_opB_{opt_b}_{s_b}"
    file_a = f"{cfg.mode.mode}mode_run_optimizer_{opt_a}_seed{s_a}"
    file_b = f"{cfg.mode.mode}mode_run_optimizer_{opt_b}_seed{s_b}"
    logger.info(f"computing {cfg.state} similarities between {opt_a}_seed_{s_a} and {opt_b}_seed_{s_b} with metric {cfg.sim_metric}")

    if cfg.state == "final":
        logger.info(f"working on final state...")
        path_a = Path(f"experiments/{file_a}/activations/final.pt")
        path_b = Path(f"experiments/{file_b}/activations/final.pt")
        actv_a = torch.load(path_a, map_location=device) # {layer_name: [n_samples, n_channels, dim_x, dim_y]}
        actv_b = torch.load(path_b, map_location=device)

        similarity = SimilarityComputer(actv_a, actv_b, experiment_name=name, model_state="final")
        similarity.compute_similarities(metric=cfg.sim_metric)
        similarity.plot_similarities(title=f"Layer similarity at model state final")
        similarity.write_similarities()

    elif cfg.state == "epoch":
        opt_path_a = Path(f"configs/optimizer/{cfg.net_1.optimizer}.yaml")
        epochs_a = OmegaConf.load(opt_path_a).epochs if cfg.mode.mode == "train" else 2
        states_a = epochs_to_states(epochs_a, 20)

        opt_path_b = Path(f"configs/optimizer/{cfg.net_2.optimizer}.yaml")
        epochs_b = OmegaConf.load(opt_path_b).epochs if cfg.mode.mode == "train" else 2
        states_b = epochs_to_states(epochs_b, 20)

        for n, s in enumerate(states_a):
            logger.info(f"working on state number {n}/20...")
            path_a = Path(f"experiments/{file_a}/activations/{s}.pt")
            path_b = Path(f"experiments/{file_b}/activations/{states_b[n]}.pt")
            actv_a = torch.load(path_a, map_location=device) # {layer_name: [n_samples, n_channels, dim_x, dim_y]}
            actv_b = torch.load(path_b, map_location=device)

            similarity = SimilarityComputer(actv_a, actv_b, experiment_name=name, model_state=n)
            similarity.compute_similarities(metric=cfg.sim_metric) 
            similarity.plot_similarities(title=f"Layer similarity at model state epoch_fraction {n}/20")
            similarity.write_similarities(output_folder="outputs/similarities")

if __name__ == "__main__":
    main()