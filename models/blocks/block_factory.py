from typing import Type

from .baseline import BaselineBlock
from .plain import PlainBlock
from .gated import GatedBlock


class BlockFactory:

    _registry = {
        'baseline':    BaselineBlock,
        'no_residual': PlainBlock,
        'gated':       GatedBlock,
    }

    @classmethod
    def get(cls, variant: str) -> Type:
        if variant not in cls._registry:
            raise ValueError(
                f"Unknown block variant='{variant}'. "
                f"Choose from {sorted(cls._registry)}"
            )
        return cls._registry[variant]

    @classmethod
    def registry(cls) -> dict:
        return cls._registry.copy()


def get_block(variant: str) -> Type:
    return BlockFactory.get(variant)


BLOCK_REGISTRY = BlockFactory.registry()
