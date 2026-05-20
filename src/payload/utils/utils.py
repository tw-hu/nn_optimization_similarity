import os
import random

from pathlib import Path
from datetime import datetime

import numpy as np
import torch

def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_time(output_dir: str):
    now = datetime.now()
    timestamp_str = now.strftime("%Y-%m-%d_%H-%M-%S")
    path = Path(output_dir) / "timestamp.txt"
    with open(path, "w") as file:
        file.write(f"File created on {timestamp_str}")