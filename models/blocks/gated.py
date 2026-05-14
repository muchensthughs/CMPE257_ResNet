import torch
import torch.nn as neural_net

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
        self.gate_conv = neural_net.Conv2d(
            BLOCK_CHANNELS, BLOCK_CHANNELS,
            kernel_size=1, stride=1, padding=0, bias=True,
        )
        # Highway-style init: gate ≈ sigmoid(3) ≈ 0.95 at start, so the
        # block behaves like baseline residual at init
        neural_net.init.zeros_(self.gate_conv.weight)
        neural_net.init.constant_(self.gate_conv.bias, 3.0)

    def _apply_shortcut(self, tensor: torch.Tensor, identity: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate_conv(identity))
        return identity + gate * tensor
