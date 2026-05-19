"""Trainer class

"""
import logging
import time
from pathlib import Path
from collections.abc import Callable

import torch
from torch import nn
from torch.amp import GradScaler, autocast
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from .optim import accuracy

logger = logging.getLogger(__name__)

class ConvTrainer:
    """Trainer class handles training and saving models
    
    Keyword arguments:
    amp -- use automatic mixed precision for computing gradient (default: True)
    save_every -- interval (epochs) at which the model and optimizer weights are saved (default: 1)
    on_epoch_end -- execute custom logic after each epoch, used for logging (default: None)
    """
    def __init__(
            self,
            model: nn.Module,
            optimizer: Optimizer,
            scheduler: LRScheduler | None,
            train_loader: DataLoader,
            val_loader: DataLoader,
            device: torch.device,
            epochs: int,
            output_dir: Path,
            amp: bool = True,
            log_every: int = 50,
            save_every: int = 5,
            on_epoch_end: Callable[[int, dict[str, float]], None] | None = None
        ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = epochs
        self.output_dir = Path(output_dir)
        self.amp = amp and device.type == "cuda"
        self.scaler = GradScaler(enabled=self.amp)
        self.criterion = nn.MSELoss()
        self.log_every = log_every
        self.save_every = save_every
        self.on_epoch_end = on_epoch_end

    def fit(self):
        # best_top1 = 0.0 # tracks best accuracy
        for epoch in range(self.epochs):
            train_metrics = self._train_one_epoch(epoch)
            val_metrics = self._evaluate()
            if self.scheduler is not None:
                self.scheduler.step()
            
            metrics = {
                **train_metrics,
                **val_metrics,
                "epoch": float(epoch),
                "lr": self.optimizer.param_groups[0]["lr"]
            }
            logger.info("epoch %d: %s", epoch, metrics)
            if self.on_epoch_end is not None:
                self.on_epoch_end(epoch, metrics)
            if epoch % self.save_every:
                self._save(f"epoch_{epoch}.pt", self.output_dir / "intermediate", epoch, metrics)
        
        self._save("final.pt", self.output_dir, self.epochs - 1, metrics)
        return metrics

    def _train_one_epoch(self, curr_epoch: int):
        """single instance of forward and backward pass"""

        self.model.train()
        total_loss = 0.0
        n = 0
        t0 = time.time()
        for step, (x, _) in enumerate(self.train_loader):
            x = x.to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=self.device.type, enabled=self.amp):
                x_recon = self.model(x)
                loss = self.criterion(x_recon, x)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            batch_size = x.size(0)
            total_loss += loss.item() * batch_size
            n += batch_size
            if step % self.log_every == 0:
                logger.info(
                    "epoch %d step %d loss=%.4f lr %.4g",
                    curr_epoch,
                    step,
                    loss.item(),
                    self.optimizer.param_groups[0]["lr"]
                )
        
        return {
            "train/loss": total_loss / max(n, 1),
            "train/epoch_time": time.time() - t0
        }
    
    @torch.no_grad()
    def _evaluate(self):
        """evaluates model and returns validation loss"""

        self.model.eval()
        total_loss = 0.0
        n = 0
        for x, _ in self.val_loader:
            x = x.to(self.device, non_blocking=True)

            with autocast(device_type=self.device.type, enabled=self.amp):
                x_recon = self.model(x)
                loss = self.criterion(x_recon, x)
            batch_size = x.size(0)
            total_loss += loss.item() * batch_size
            n += batch_size
        
        return{
            "val/loss": total_loss / max(n, 1)
        }

    def _save(self, name: str, dir: str | Path, epoch: int, metrics: dict[str, float]):
        """saves model weights and trainer state"""
        # path = self.output_dir / name
        dir.mkdir(parents=True, exist_ok=True)
        path = Path(dir) / name

        torch.save(
            {
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "scheduler_state": self.scheduler.state_dict() if self.scheduler else None,
                "epoch": epoch,
                "metrics": metrics
            },
            path
        )
        logger.info("saved model -> %s", path)