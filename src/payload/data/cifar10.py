"""CIFAR-10 data loading and visualization

"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Subset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

# CIFAR means and standard deviations in (R, G, B) channels
CIFAR_MEAN = (0.4914, 0.4822 ,0.4465) 
CIFAR_STD = (0.2470, 0.2435, 0.2616)
CIFAR_CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def cifar_transform():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
        ]
    )

def build_cifar(
        root: str | Path,
        split: str = "train",
        transform = cifar_transform(),
        mode: str = "dev",
        test_fraction: int = 0.01
        ):
    root = Path(root) / split
    if not root.exists():
        raise FileNotFoundError(
            f"CIFAR-10 split not found at {root}. Expected ImageFolder layout."
        )
    dataset = ImageFolder(str(root), transform = transform)
    if mode == "dev":
        num_fraction = int(len(dataset) * test_fraction)
        idx = np.random.choice(len(dataset), num_fraction, replace=False)
        return Subset(dataset, idx)
    else:
        return dataset

def build_dataloader(dataset: Dataset, batch_size: int, num_workers: int = 1, seed: int = 0):
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, generator=g)