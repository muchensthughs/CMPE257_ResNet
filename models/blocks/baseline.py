import torch
import torch.nn as nn
from typing import Optional

from .base import ResidualBlockBase, conv3x3


class BaselineBlock(ResidualBlockBase):
    """
    Standard residual block.

    Shortcut is identity or a learned projection when dimensions change.

    Forward: out = ReLU(F(x) + shortcut(x))
    """

    def _apply_shortcut(self, out: torch.Tensor, identity: torch.Tensor) -> torch.Tensor:
        return out + identity
