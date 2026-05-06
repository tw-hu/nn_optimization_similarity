"""Downloads and organizes CIFAR-10 Dataset into ImageFolder format

"""

from pathlib import Path

import numpy as np
import torch
import torchvision
from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset, Subset

# CIFAR-10 has 50000 images
NUM_TRAIN = 40000
NUM_VAL = 10000

DATA_DIR = "./data"

SEED = 0

def save_split(data: Dataset | Subset, root: str | Path, split_name: str):
    if split_name not in ["train", "val", "test"]:
        raise ValueError("split_name must be 'train', 'val', or 'test'")
    
    if isinstance(data, Subset):
        dataset = data.dataset
    else:
        dataset = data  
    classes = dataset.classes

    root = Path(root) / split_name

    for i, (img, label_idx) in enumerate(data):
        class_name = classes[label_idx]
        class_path = root / class_name
        class_path.mkdir(parents = True, exist_ok = True)
        img_path = class_path / f"{i}.png"
        img.save(img_path)

def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    data_dir = Path(DATA_DIR)
    if not data_dir.exists():
        raise FileNotFoundError(
            f"./data folder not found. Are you running the script in root folder?"
        )
    
    full_train_dataset = torchvision.datasets.CIFAR10(root = data_dir, train = True, download = True)
    test_dataset = torchvision.datasets.CIFAR10(root = data_dir, train = False, download = True)

    indices = np.random.permutation(len(full_train_dataset))
    train_idx = indices[:NUM_TRAIN]
    val_idx = indices[NUM_TRAIN:]

    train_dataset = Subset(full_train_dataset, train_idx)
    val_dataset = Subset(full_train_dataset, val_idx)

    save_split(train_dataset, data_dir, "train")
    save_split(val_dataset, data_dir, "val")
    save_split(test_dataset, data_dir, "test")

if __name__ == "__main__":
    main()