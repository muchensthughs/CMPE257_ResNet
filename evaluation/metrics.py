from typing import List, Tuple

import torch


class AverageMeter:
    """Tracks a running mean of a scalar metric."""

    def __init__(self, name: str):
        self.name = name
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.sum += val * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count else 0.0


def accuracy(output: torch.Tensor, target: torch.Tensor,
             topk: Tuple[int, ...] = (1,)) -> List[torch.Tensor]:
    """Return top-k accuracy values for each k in topk."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        results = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum()
            results.append(correct_k.mul_(100.0 / batch_size))
        return results
