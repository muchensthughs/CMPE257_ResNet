"""
models/backbone.py
==================
Generic ResNet backbone for shallow CIFAR networks with configurable block variants.

Depth configurations
--------------------
CIFAR: 20, 32, 44, 56, 110  (3 stages, 32×32 inputs)

The block variant is injected at construction, so the backbone itself is
architecture-agnostic and purely responsible for stage/layer assembly.
"""

import torch
import torch.nn as nn
from typing import List, Optional, Type, Union

from .blocks import get_block, make_downsample


# ---------------------------------------------------------------------------
# Depth → stage layout tables
# ---------------------------------------------------------------------------

# CIFAR: depth = 6n + 2 where n = blocks per stage
_CIFAR_STAGE_BLOCKS = {
    8:  [1, 1, 1],
    20: [3, 3, 3],
}


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------

class ResNetBackbone(nn.Module):
    """
    Parameterised ResNet backbone for shallow CIFAR networks.

    Args:
        variant         : 'baseline' | 'no_residual' | 'scaled' | 'gated'
        depth           : total network depth (20, 32, 44, 56, 110)
        num_classes     : output dimension
        width_factor    : multiplier on base channel widths (1.0 = standard)
        block_kwargs    : extra kwargs forwarded to every block (e.g. alpha,
                          learnable_alpha, gate_reduction)
    """

    def __init__(
        self,
        variant: str = 'baseline',
        depth: int = 20,
        num_classes: int = 10,
        width_factor: float = 1.0,
        **block_kwargs,
    ):
        super().__init__()

        self.variant   = variant
        self.depth     = depth

        if depth not in _CIFAR_STAGE_BLOCKS:
            raise ValueError(f"CIFAR depth must be one of {list(_CIFAR_STAGE_BLOCKS)}")
        
        stage_blocks = _CIFAR_STAGE_BLOCKS[depth]
        base_widths  = [int(w * width_factor) for w in [16, 32, 64]]

        block_cls   = get_block(variant)
        self.in_channels = base_widths[0]

        # ── Stem ──────────────────────────────────────────────────────────
        self.stem = nn.Sequential(
            nn.Conv2d(3, base_widths[0], kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_widths[0]),
            nn.ReLU(inplace=True),
        )
        strides = [1, 2, 2]

        # ── Stages ────────────────────────────────────────────────────────
        self.stages = nn.ModuleList()
        for out_channels, n_blocks, stride in zip(base_widths, stage_blocks, strides):
            self.stages.append(
                self._make_stage(block_cls, out_channels, n_blocks, stride,
                                 **block_kwargs)
            )

        # ── Head ──────────────────────────────────────────────────────────
        final_channels = base_widths[-1]
        self.avgpool   = nn.AdaptiveAvgPool2d(1)
        self.fc        = nn.Linear(final_channels, num_classes)

        self._init_weights()

    def _make_stage(self, block_cls, out_channels, n_blocks, stride,
                    **block_kwargs):
        """Build one stage (sequence of blocks)."""
        layers = []

        for i in range(n_blocks):
            s = stride if i == 0 else 1
            downsample = make_downsample(self.in_channels, out_channels, s)
            layers.append(
                block_cls(self.in_channels, out_channels, stride=s,
                          downsample=downsample,
                          **block_kwargs)
            )
            self.in_channels = out_channels

        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

    # ── Convenience properties ─────────────────────────────────────────────

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

