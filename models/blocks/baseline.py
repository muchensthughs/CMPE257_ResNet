import torch

from .base import ResidualBlockBase


class BaselineBlock(ResidualBlockBase):
    """
    Baseline residual block: y = F(x) + x

    Inherits the shared F(x) pathway from ResidualBlockBase.
    """

    def _apply_shortcut(self, tensor: torch.Tensor, identity: torch.Tensor) -> torch.Tensor:
        return tensor + identity
