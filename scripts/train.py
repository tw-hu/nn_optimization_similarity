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

from payload.data import build_convnet, make_loader
from payload.training import Trainer, build_optimizer, build_scheduler

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    main()