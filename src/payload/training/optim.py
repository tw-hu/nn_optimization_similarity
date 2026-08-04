"""Simple scripts used for training"""

import torch
from torch import nn
from torch.optim import Adam, AdamW, SGD
from adam_atan2_pytorch import AdamAtan2

def build_optimizer(
        model: nn.Module,
        optimizer: str,
        lr: float,
        weight_decay: float = 5.0e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        momentum: float = 0.9,
        nesterov: bool = True,
        skip_list: tuple = (nn.BatchNorm2d)
):
    """
    Simple method to exclude BatchNorm2D layers and biases from weight decay
    """
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias") or any(isinstance(m, skip_list) for m in model.modules()):
            no_decay.append(param)
        else:
            decay.append(param)
    hypers = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0}
    ]
    
    if optimizer == "adam_atan2":
        return AdamAtan2(hypers, lr=lr, betas=betas)
    elif optimizer == "adamw":
        return AdamW(hypers, lr=lr, betas=betas)
    elif optimizer == "adam":
        return Adam(hypers, lr=lr, betas=betas)
    elif optimizer == "sgd":
        return SGD(hypers, lr=lr, momentum=momentum, nesterov=nesterov)

@torch.no_grad()
def accuracy(preds: torch.Tensor, labels: torch.Tensor, topk: tuple[int] = (1, 5)):
    max_k = max(topk)
    _, indices = preds.topk(max_k, dim=1, largest=True, sorted=True)
    correct = indices.eq(labels.view(-1, 1).expand_as(indices))
    return [float(correct[:, :k].any(dim=1).float().mean().item()) for k in topk]
