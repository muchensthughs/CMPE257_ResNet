import torch
import torch.nn as nn
from typing import Optional


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def conv3x3(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding."""
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution."""
    return nn.Conv2d(in_channels, out_channels, kernel_size=1,
                     stride=stride, bias=False)


def make_downsample(in_channels: int, out_channels: int,
                    stride: int) -> Optional[nn.Sequential]:
    """Projection shortcut when dimension changes."""
    if stride != 1 or in_channels != out_channels:
        return nn.Sequential(
            conv1x1(in_channels, out_channels, stride),
            nn.BatchNorm2d(out_channels),
        )
    return None


# ---------------------------------------------------------------------------
# Base class for residual blocks
# ---------------------------------------------------------------------------

class ResidualBlockBase(nn.Module):
    """
    Base class for all residual blocks.

    Handles common initialization and the shared forward pass up to the shortcut.
    Subclasses implement _apply_shortcut() to customize the connection.

    Body (identical for all variants):
        out = ReLU(BN(Conv3x3(x)))
        out = BN(Conv3x3(out))
        (optional attention module)

    Then subclasses define how the shortcut is applied to produce the final output.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 downsample: Optional[nn.Module] = None):
        super().__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def _body(self, x: torch.Tensor) -> torch.Tensor:
        """Common body: Conv → BN → ReLU → Conv → BN → (optional attention)."""
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return out

    def _get_identity(self, x: torch.Tensor) -> torch.Tensor:
        """Get the identity/shortcut, applying downsample if needed."""
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        return identity

    def _apply_shortcut(self, out: torch.Tensor, identity: torch.Tensor) -> torch.Tensor:
        """Apply shortcut connection. Override in subclasses."""
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self._body(x)
        identity = self._get_identity(x)
        out = self._apply_shortcut(out, identity)
        return self.relu(out)
