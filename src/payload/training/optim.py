import torch

@torch.no_grad()
def accuracy(preds: torch.Tensor, labels: torch.Tensor, topk: tuple[int] = (1, 5)):
    max_k = max(topk)
    _, indices = preds.topk(max_k, dim=1, largest=True, sorted=True)
    correct = indices.eq(labels.view(-1, 1).expand_as(indices))
    return [float(correct[:, :k].any(dim=1).float().mean().item()) for k in topk]