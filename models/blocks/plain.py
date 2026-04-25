import torch

from .base import ResidualBlockBase


class PlainBlock(ResidualBlockBase):
    """
    Plain (no residual) block: y = F(x)

    Strictly no skip connection of any kind (paper section 4).
    Serves as the ablation baseline to expose the degradation problem.
    """

    def _apply_shortcut(self, out: torch.Tensor, identity: torch.Tensor) -> torch.Tensor:
        return out
