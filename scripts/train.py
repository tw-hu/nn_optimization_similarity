"""Train a single instance of a convolutional autoencoder on CIFAR-10

Usage:

1. Configure configs/config.yaml file
2. Single run:
    python scripts/train.py
3. Multi run:
    python scripts/train.py -m seed=0,1,2,3,4
"""
import logging
from pathlib import Path

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
import hydra
import wandb
from omegaconf import DictConfig, OmegaConf

from payload.data.cifar10 import build_cifar, build_dataloader
from payload.training.ConvTrainer import ConvTrainer
from payload.training.optim import build_optimizer
from payload.models.ConvClassifier import build_model
from payload.utils.utils import set_seed, save_time

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../configs", config_name="config")
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
    train_set = build_cifar(Path(cfg.data_dir), "train", mini=cfg.mode.mini)
    train_loader = build_dataloader(
        train_set,
        batch_size=cfg.optimizer.batch_size,
        num_workers=cfg.num_workers,
        seed=cfg.seed,
        pin_memory=(cfg.device == "cuda" and torch.cuda.is_available()))

    val_set = build_cifar(Path(cfg.data_dir), "val", mini=cfg.mode.mini)
    val_loader = build_dataloader(
        val_set,
        batch_size=cfg.optimizer.batch_size,
        num_workers=cfg.num_workers,
        seed=cfg.seed,
        pin_memory=(cfg.device == "cuda" and torch.cuda.is_available()))

    # Model + optimizer
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    model = build_model()
    
    optimizer = build_optimizer(
        model,
        optimizer=cfg.optimizer.name,
        lr=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=(cfg.optimizer.epochs if cfg.mode=="train" else 2))

    # Training loop
    def on_epoch_end(epoch: int, metrics: dict) -> None:
        """calls logger on epoch end"""
        if wandb_run is not None:
            wandb.log(metrics, step=epoch)

    trainer = ConvTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=(cfg.optimizer.epochs if cfg.mode=="train" else 2),
        output_dir=output_dir,
        amp=cfg.mode.training.amp,
        log_every=cfg.mode.training.log_every,
        on_epoch_end=on_epoch_end
    )
    metrics = trainer.fit()
    save_time(output_dir)

    if wandb_run is not None:
        wandb.summary.update(metrics)
        wandb.finish()
    return float(metrics["val/loss"])

if __name__ == "__main__":
    main()