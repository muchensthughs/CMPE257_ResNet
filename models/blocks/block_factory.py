"""
models/blocks/block_factory.py
==============================
Block factory for creating and managing residual block variants.
"""

from typing import Type

from .baseline import BaselineBlock


class BlockFactory:
    """
    Factory for creating and managing residual block variants.
    
    Provides a centralized registry of available block types and methods
    to retrieve them by name.
    """
    
    _registry = {
        'baseline':    BaselineBlock
    }
    
    @classmethod
    def get(cls, variant: str) -> Type:
        if variant not in cls._registry:
            raise ValueError(
                f"Unknown block variant='{variant}'. "
            )
        return cls._registry[variant]
    
    @classmethod
    def registry(cls) -> dict:
        """Return the full registry as a dict."""
        return cls._registry.copy()


# Backward compatibility: maintain get_block function and BLOCK_REGISTRY
def get_block(variant: str) -> Type:
    """Return the block class for the requested variant."""
    return BlockFactory.get(variant)


BLOCK_REGISTRY = BlockFactory.registry()