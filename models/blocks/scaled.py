import torch
import torch.nn as nn
from .base import ResidualBlockBase, BLOCK_CHANNELS


class ScaledBlock(ResidualBlockBase):
    """
    Scaled residual block: y = x + α · F(x)

    α is a per-channel learnable parameter (shape: [64]) initialized to 1.0,
    so each channel's residual contribution is scaled independently. At init
    this block is equivalent to baseline. The skip connection (x) is passed
    through unscaled. Inherits the shared F(x) pathway from ResidualBlockBase.
    """

    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(BLOCK_CHANNELS))

    def _apply_shortcut(self, out: torch.Tensor, identity: torch.Tensor) -> torch.Tensor:
        return identity + self.alpha.view(1, -1, 1, 1) * out

