"""Script to extract activations from trained neural networks

Usage:
python scripts/extract_activations.py path_to_pt
"""

from pathlib import Path
import logging

import torch
import hydra
from omegaconf import DictConfig

from payload.analysis.ActivationCollector import ActivationCollector
from payload.data.cifar10 import build_cifar
from payload.models.ConvClassifier import build_model
from payload.utils.utils import set_seed

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    set_seed(10)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    dir = Path("experiments") / cfg.experiment_name
    output_dir = dir / "activations"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("building probe dataset")
    probeset = build_cifar(Path(cfg.data_dir), "val", mini=True)

    logger.info(f"collecting activations of final model")
    model = build_model()
    collector = ActivationCollector(
        model,
        model_name=f"{cfg.experiment_name}_final",
        dataset=probeset,
        device=device,
        num_samples=128
        )
    collector.import_pt(dir / f"final.pt")
    actvs = collector.compute_activations() # actvs = {layer_name: [n_samples, n_channels, dim_x, dim_y]}
    torch.save(actvs, output_dir / f"final.pt")

    if cfg.mode.intermediate:
        for epoch in range(0, (cfg.optimizer.epochs if cfg.mode == "train" else 2), 5):
            logger.info(f"collecting activations of model at epoch {epoch}")
            model = build_model()
            collector = ActivationCollector(
                model,
                model_name=f"{cfg.experiment_name}_epoch_{epoch}",
                dataset=probeset,
                device=device,
                num_samples=128
                )
            collector.import_pt(dir / f"epoch_{epoch}.pt")
            actvs = collector.compute_activations() # actvs = {layer_name: [n_samples, n_channels, dim_x, dim_y]}
            torch.save(actvs, output_dir / f"epoch_{epoch}.pt")

if __name__ == "__main__":
    main()
