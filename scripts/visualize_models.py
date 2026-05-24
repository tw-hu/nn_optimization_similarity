"""
Script which creates model, loads dataset, loads weights into model and visualizes output
"""
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import hydra
from omegaconf import OmegaConf, DictConfig

from payload.data.cifar10 import build_cifar, cifar_inverse_transform
from payload.models.ConvClassifier import build_model
from payload.analysis.ModelVisualizer import ModelVisualizer
from payload.utils.utils import set_seed

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    set_seed(cfg.seed)
    test_set = build_cifar(Path(cfg.data_dir), "test", mini=cfg.mode.mini)

    model = build_model()
    visualizer = ModelVisualizer(
        model,
        model_name=f"{cfg.experiment_name}_{cfg.file_pt}",
        num_samples=16
    )
    dir = Path("experiments") / cfg.experiment_name
    visualizer.import_pt(dir / f"{cfg.file_pt}.pt")
    visualizer.forward_data(test_set)

    output_dir = Path(f"outputs/plots/{cfg.experiment_name}_{cfg.file_pt}.png")
    visualizer.generate_plots(output_dir=output_dir)

if __name__ == "__main__":
    main()