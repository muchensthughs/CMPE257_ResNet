from typing import Type

from .baseline import BaselineBlock
from .plain import PlainBlock
from .gated import GatedBlock
from .scaled import ScaledBlock


class BlockFactory:
    """
    Maps the `model.variant` config string to its residual-block class.

    The backbone is shared across experiments; only the block swaps out.
    Add a new variant by importing its class and registering it below.
    """

    # Note: 'no_residual' maps to PlainBlock (drops the identity shortcut).
    _registry = {
        'baseline':    BaselineBlock,
        'no_residual': PlainBlock,
        'gated':       GatedBlock,
        'scaled':      ScaledBlock,
    }

    @classmethod
    def get(cls, variant: str) -> Type:
        """Return the block class for `variant`, or raise if unknown."""
        if variant not in cls._registry:
            raise ValueError(
                f"Unknown block variant='{variant}'. "
                f"Choose from {sorted(cls._registry)}"
            )
        return cls._registry[variant]

    @classmethod
    def registry(cls) -> dict:
        # Return a copy so callers can't mutate the source of truth.
        return cls._registry.copy()


def get_block(variant: str) -> Type:
    """Convenience wrapper around BlockFactory.get."""
    return BlockFactory.get(variant)


BLOCK_REGISTRY = BlockFactory.registry()
