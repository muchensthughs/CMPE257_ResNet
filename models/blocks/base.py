import torch
import torch.nn as nn

FILTERS = 64


def conv3x3(in_channels: int, out_channels: int) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=1, padding=1, bias=False)


class ResidualBlockBase(nn.Module):
    """
    Shared F(x) pathway required by the paper:
        Conv2d(64) -> BN -> ReLU -> Conv2d(64) -> BN  =  F(x)

    Subclasses implement _apply_shortcut(F(x), identity) -> y.
    forward() returns ReLU(y).
    """

    def __init__(self):
        super().__init__()
        self.conv1 = conv3x3(FILTERS, FILTERS)
        self.bn1   = nn.BatchNorm2d(FILTERS)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(FILTERS, FILTERS)
        self.bn2   = nn.BatchNorm2d(FILTERS)

    def _body(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return out

    def _apply_shortcut(self, out: torch.Tensor, identity: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self._body(x)
        out = self._apply_shortcut(out, identity)
        return self.relu(out)
