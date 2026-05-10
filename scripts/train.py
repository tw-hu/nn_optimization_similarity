"""Train a single instance of a shallow CNN on CIFAR-10

Usage:

Single run:
    python scripts/train.py training.lr=0.05 training.beta1=0.05 training.beta2=0.05

Multi-run:
    python scripts/train.py -m\
"""
import json
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader, ImageFolder
import hydra
import wandb
from omegaconf import DictConfig, OmegaConf

from payload.data import build_convnet, make_loader
from payload.training import Trainer, build_optimizer, build_scheduler, cifar_transform, accuracy
from payload.models import ConvClassifier, smallCNN, largeCNN, build_mlp
from payload.utils import set_seed

logger = logging.getLogger(__name__)

def main(cfg: DictConfig):
    set_seed(cfg.seed)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Logging (weights and biases)
    wandb_run = None
    if cfg.wandb.enabled:
        wandb_run = wandb.init(
            project=cfg.wandb.project,
            mode=cfg.wandb.mode,
            name=cfg.experiment_name,
            dir=str(output_dir),
            config=OmegaConf.to_container(cfg, resolve=True)
        )

    # Data
    train_set = ImageFolder(Path(cfg.data_dir) / "train", transform=cifar_transform)
    train_loader = DataLoader(train_set, batch_size=cfg.training.batch_size, seed=cfg.seed)

    val_set = ImageFolder(Path(cfg.data_dir) / "val", transform=cifar_transform)
    val_loader = DataLoader(val_set, batch_size=cfg.training.batch_size, seed=cfg.seed)

    # Model + optimizer
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    act_kwargs = {"inplace": True}
    conv_model = smallCNN(activation="relu", **act_kwargs)
    fc_model = build_mlp([1024, 10])
    model = ConvClassifier(
        conv_model,
        fc_model
    )

    # Training loop
    def on_epoch_end(epoch: int, metrics: dict) -> None:
        if wandb_run is not None:
            wandb.log(metrics, step=epoch)

    trainer = Trainer(
        model=model,
        optimizer=optimizer, # code this
        scheduler=scheduler, # code this
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=cfg.epochs,
        output_dir=output_dir,
        amp=cfg.training.amp,
        log_every=cfg.log_every,
        on_epoch_end=on_epoch_end
    )
    metrics = trainer.fit()

    # Save models

if __name__ == "__main__":
    main()