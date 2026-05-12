import torch
import torch.nn as nn
from .base import ResidualBlockBase


class ScaledBlock(ResidualBlockBase):
    """
    Scaled residual block: y = x + α · F(x)

    α is a single learnable scalar initialized to 1.0, so at init this block
    is equivalent to baseline. The scalar broadcasts over [B, C, H, W], scaling
    F(x) uniformly across channels and spatial locations. The skip connection
    (x) is passed through unscaled. Inherits the shared F(x) pathway from
    ResidualBlockBase.
    """

    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def _apply_shortcut(self, out: torch.Tensor, identity: torch.Tensor) -> torch.Tensor:
        return identity + self.alpha * out

