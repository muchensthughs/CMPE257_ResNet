import torch
import torch.nn as neural_net

# Number of convolutional filters (output channels) used by every Conv2d in
# every block, for every variant, at every depth. Locked at 64 per the variants
# paper §6.2.1 so that block capacity is identical across runs — any observed
# difference in accuracy or gradient flow is then attributable solely to the
# skip-connection mechanism, not to channel-width changes.
BLOCK_CHANNELS = 64


def conv3x3(in_channels: int, out_channels: int) -> neural_net.Conv2d:
    return neural_net.Conv2d(in_channels, out_channels, kernel_size=3,
                             stride=1, padding=1, bias=False)


class ResidualBlockBase(neural_net.Module):
    """
    Shared F(x) pathway required by the paper:
        Conv2d(64) -> BN -> ReLU -> Conv2d(64) -> BN  =  F(x)

    Subclasses implement _apply_shortcut(F(x), identity) -> y.
    forward() returns ReLU(y).
    """

    def __init__(self):
        super().__init__()
        self.conv1 = conv3x3(BLOCK_CHANNELS, BLOCK_CHANNELS)
        self.bn1   = neural_net.BatchNorm2d(BLOCK_CHANNELS)
        self.relu  = neural_net.ReLU(inplace=True)
        self.conv2 = conv3x3(BLOCK_CHANNELS, BLOCK_CHANNELS)
        self.bn2   = neural_net.BatchNorm2d(BLOCK_CHANNELS)

    def _body(self, tensor: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(tensor)))
        out = self.bn2(self.conv2(out))
        return out

    """
    This function should be overriden to implement the specific skip connection logic for each block variant.
    """
    def _apply_shortcut(self, out: torch.Tensor, identity: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        identity = tensor
        out = self._body(tensor)
        out = self._apply_shortcut(out, identity)
        return self.relu(out)
