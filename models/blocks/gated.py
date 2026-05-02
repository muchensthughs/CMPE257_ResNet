import torch
import torch.nn as nn

from .base import ResidualBlockBase, BLOCK_CHANNELS


class GatedBlock(ResidualBlockBase):
    """
    Gated residual block: y = x + g(x) * F(x)

    The gate g(x) = sigmoid(W * x) is computed by a 1x1 convolution on the
    block input, producing a tensor with the same [B, C, H, W] shape as F(x).
    Element-wise multiplication then modulates the residual contribution
    per-channel and per-spatial-location.

    Inherits the shared F(x) pathway from ResidualBlockBase.
    """

    def __init__(self):
        super().__init__()
        # 1x1 conv: bias=True since this gate is not followed by BatchNorm
        self.gate_conv = nn.Conv2d(
            BLOCK_CHANNELS, BLOCK_CHANNELS,
            kernel_size=1, stride=1, padding=0, bias=True,
        )

    def _apply_shortcut(self, out: torch.Tensor, identity: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate_conv(identity))
        return identity + gate * out
