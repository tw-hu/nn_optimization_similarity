"""
Lightweight class for visualizing models
"""
import os
import csv

import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from payload.data.cifar10 import cifar_inverse_transform

STATES = [f"epoch_{n}" for n in range(0, 101, 5)]

class ModelVisualizer():
    """
    A class which handles i/o of model weights, sets up hooks, and analyzes activations
    """
    def __init__(
            self,
            model: nn.Module,
            device: str = "cpu",
            num_samples: int = 16
    ):
        self.model = model
        self.device = device
        self.num_samples = num_samples
        self.experiment = None
        self.dataset = None

    def load_experiment(self, input_dir: str | Path) -> None:
        self.experiment = input_dir

    def load_dataset(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def generate_reconstructions(self, output_dir: str | Path, state: str = "epoch_100") -> None:
        self._load_pt(state)
        xs, xs_rec = self._forward_data()

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        # Function to plot a 4x4 grid on a specific axis
        def _plot_4x4_grid(ax, images, title):
            ax.axis('off')
            ax.set_title(title, fontsize=14, pad=15)
            
            # Create an empty canvas to stitch the 16 images together into one large 4x4 image
            # (4 images * 32 pixels) = 128 pixels wide/high
            grid = np.zeros((4 * 32, 4 * 32, 3))
            
            idx = 0
            for r in range(4):
                for c in range(4):
                    x = c * 32
                    y = r * 32
                    grid[y:y+32, x:x+32, :] = images[idx]
                    idx += 1
                    
            ax.imshow(grid)

        xs = np.clip(xs, 0.0, 1.0)
        xs_rec = np.clip(xs_rec, 0.0, 1.0)
        
        _plot_4x4_grid(axes[0], np.transpose(xs, (0, 2, 3, 1)), "Original CIFAR-10 Images")
        _plot_4x4_grid(axes[1], np.transpose(xs_rec, (0, 2, 3, 1)), f"Image Reconstructions")

        save_file = output_path / "reconstructions.png"
        plt.savefig(save_file)
        plt.close()

    def generate_loss_plot(self, output_dir: str | Path) -> None:
        train_loss, val_loss = self._get_losses()

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        epochs = [n for n in range(5, 101, 5)]

        csv_file = output_path / "losses.csv"
        with open(csv_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_loss"])
            for n in range(len(epochs)):
                writer.writerow([epochs[n], train_loss[n], val_loss[n]])

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_loss, label="Train Loss", color="blue", marker="o", linestyle="-")
        plt.plot(epochs, val_loss, label="Validation Loss", color="orange", marker="s", linestyle="--")

        plt.title("Training and Validation Loss", fontsize=14, pad=15)
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.legend(loc="upper right", fontsize=11)
        plt.grid(True, linestyle=":", alpha=0.7)
        
        save_file = output_path / "loss_plot.png"
        plt.savefig(save_file, dpi=300, bbox_inches="tight")
        plt.close()

    def _forward_data(self) -> list:
        if self.dataset is None:
            raise ValueError("select an dataset with load_dataset")
        img_loader = DataLoader(self.dataset, batch_size=self.num_samples, shuffle=True)
        xs, _ = next(iter(img_loader)) # torch.Size([num_samples, 3, 32, 32])
        xs_rec = cifar_inverse_transform()(self.model(xs))
        xs_rec = xs_rec.detach().numpy()
        xs = cifar_inverse_transform()(xs).detach().numpy()
        return xs, xs_rec

    def _get_metrics(self, input_dir: str | Path) -> dict:
        pt_dict = torch.load(Path(input_dir), map_location=self.device)
        return pt_dict["metrics"]

    def _get_losses(self) -> tuple[list, list]:
        train_loss, val_loss = [], []
        if self.experiment is None:
            raise ValueError("select an experiment file with load_experiment")
        for n, state in enumerate(STATES):
            if n == 0: continue
            file_dir = self.experiment / f"{state}.pt"
            metrics = self._get_metrics(file_dir)
            train_loss.append(metrics["train/loss"])
            val_loss.append(metrics["val/loss"])
        return train_loss, val_loss

    def _load_pt(self, state: str) -> None:
        pt_dict = torch.load(self.experiment / f"{state}.pt", map_location=self.device)
        model_weights = pt_dict["model_state"]
        self._assert_state_dict(self.model, model_weights)
        self.model.load_state_dict(model_weights)

    def _assert_state_dict(self, model, state_dict) -> None:
        model_state = model.state_dict()
        for key, value in state_dict.items():
            if key in model_state:
                assert value.shape == model_state[key].shape, \
                    f"Shape mismatch for {key}: checkpoint {value.shape} vs model {model_state[key].shape}"
            else:
                print(f"Warning: {key} not found in model state_dict")
