import torch
import torch.nn as neural_net

from .blocks import get_block
from .blocks.base import BLOCK_CHANNELS

_VALID_DEPTHS = {4, 32, 50}


class ResNetBackbone(neural_net.Module):
    """
    Flat ResNet backbone for CIFAR-10.

    Macro-architecture (per paper section 6.2):
        Stem Conv(3->64) -> N Blocks (all 64 filters) -> Global Avg Pool -> Linear(64, 10)

    Args:
        variant     : block variant name registered in block_factory
        depth       : total number of blocks — 4, 32, or 50
        num_classes : output dimension (10 for CIFAR-10)
    """

    def __init__(self, variant: str = 'baseline', depth: int = 4, num_classes: int = 10):
        super().__init__()

        if depth not in _VALID_DEPTHS:
            raise ValueError(f"depth must be one of {_VALID_DEPTHS}, got {depth}")

        self.variant = variant
        self.depth = depth

        block_cls = get_block(variant)

        self.stem = neural_net.Sequential(
            neural_net.Conv2d(3, BLOCK_CHANNELS, kernel_size=3, stride=1, padding=1, bias=False),
            neural_net.BatchNorm2d(BLOCK_CHANNELS),
            neural_net.ReLU(inplace=True),
        )

        self.blocks = neural_net.Sequential(*[block_cls() for _ in range(depth)])

        self.avgpool = neural_net.AdaptiveAvgPool2d(1)
        self.fc = neural_net.Linear(BLOCK_CHANNELS, num_classes)

        self._init_weights()

    def _init_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, neural_net.Conv2d):
                if name.endswith('gate_conv'):
                    continue
                neural_net.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, neural_net.BatchNorm2d):
                neural_net.init.constant_(module.weight, 1)
                neural_net.init.constant_(module.bias, 0)
            elif isinstance(module, neural_net.Linear):
                neural_net.init.normal_(module.weight, 0, 0.01)
                neural_net.init.constant_(module.bias, 0)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = self.stem(tensor)
        tensor = self.blocks(tensor)
        tensor = self.avgpool(tensor)
        tensor = torch.flatten(tensor, 1)
        return self.fc(tensor)

    @property
    def num_parameters(self) -> int:
        return sum(param.numel() for param in self.parameters() if param.requires_grad)
