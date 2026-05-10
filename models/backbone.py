import torch
import torch.nn as nn

from .blocks import get_block
from .blocks.base import BLOCK_CHANNELS

_VALID_DEPTHS = {4, 32, 50}


class ResNetBackbone(nn.Module):
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

        self.stem = nn.Sequential(
            nn.Conv2d(3, BLOCK_CHANNELS, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(BLOCK_CHANNELS),
            nn.ReLU(inplace=True),
        )

        self.blocks = nn.Sequential(*[block_cls() for _ in range(depth)])

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(BLOCK_CHANNELS, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
