"""Script to extract activations from trained neural networks

Usage:
python scripts/extract_activations.py
"""

from pathlib import Path
import logging

import torch
import hydra
from omegaconf import OmegaConf, DictConfig

from payload.analysis.ActivationCollector import ActivationCollector
from payload.data.cifar10 import build_cifar
from payload.models.ConvClassifier import build_model
from payload.utils.utils import set_seed

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    set_seed(10)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    logger.info(f"working on {cfg.experiment_name}")
    dir = Path("experiments") / cfg.experiment_name
    output_dir = dir / "activations"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("building probe dataset...")
    probeset = build_cifar(Path(cfg.data_dir), "val", mini=True)

    logger.info(f"beginning acivation collection...")

    if cfg.state == "all":
        opt_path = Path(f"configs/optimizer/{cfg.optimizer.name}.yaml")
        epochs = OmegaConf.load(opt_path).epochs if cfg.mode.mode == "train" else 2
        for n in range(0, epochs + 1, 5):
            model_states.append(f"epoch_{n}")
    else:
        model_states = [cfg.state]

    for s in model_states:
        logger.info(f"working on state {s}...")
        model = build_model().to(device)
        collector = ActivationCollector(
            model,
            model_name=f"{cfg.experiment_name}_{s}",
            dataset=probeset,
            device=device,
            num_samples=128
            )
        if s != "init":
            collector.import_pt(dir / f"{s}.pt")
        actvs = collector.compute_activations() # actvs = {layer_name: [n_samples, n_channels, dim_x, dim_y]}
        torch.save(actvs, output_dir / f"{s}.pt")

if __name__ == "__main__":
    main()