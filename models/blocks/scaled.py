import torch
import torch.nn as nn
from .base import ResidualBlockBase


class ScaledBlock(ResidualBlockBase):
    """
    Scaled residual block: y = x + α · F(x)

    α is a learnable scalar parameter initialized to 1.0.
    The skip connection (x) is passed through unscaled.
    Inherits the shared F(x) pathway from ResidualBlockBase.
    """

    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1))

    def _apply_shortcut(self, out: torch.Tensor, identity: torch.Tensor) -> torch.Tensor:
        return identity + self.alpha * out
