import torch
import torch.nn as nn
from .base import ResidualBlockBase, BLOCK_CHANNELS


class ScaledBlock(ResidualBlockBase):
    """
    Scaled residual block: y = x + α · F(x)

    α is a per-channel learnable parameter (shape: [64]) initialized to 0.0
    (ReZero-style) so the network starts as pure identity mappings and learns to
    enable residual contributions. This dramatically improves gradient flow in
    deep networks — at init F(x) is gated out entirely, avoiding the compounding
    of random residual noise across many layers.
    nn.Parameter is not an nn.Module, so _init_weights() skips it automatically —
    no flag needed. torch.zeros(BLOCK_CHANNELS) is deterministic (no RNG consumed).
    The skip connection (x) is passed through unscaled.
    Inherits the shared F(x) pathway from ResidualBlockBase.
    """

    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(BLOCK_CHANNELS))

    def _apply_shortcut(self, out: torch.Tensor, identity: torch.Tensor) -> torch.Tensor:
        return identity + self.alpha.view(1, -1, 1, 1) * out
