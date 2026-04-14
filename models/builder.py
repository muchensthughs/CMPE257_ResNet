"""
models/builder.py
=================
Model building utilities. Constructs ResNetBackbone from config dicts.
"""

from .backbone import ResNetBackbone


# CIFAR-10 has 10 classes
NUM_CLASSES = 10


def build_model(cfg: dict) -> ResNetBackbone:
    """
    Instantiate a ResNetBackbone from a nested config dict.

    Expected config shape (subset of configs/base.yaml):
    {
      'model': {
        'variant': 'baseline',
        'depth': 20,
        'width_factor': 1.0,
        # variant-specific keys passed as block_kwargs:
      }
    }
    """
    mcfg    = cfg['model']
    variant = mcfg['variant']
    depth   = mcfg.get('depth', 20)

    # Collect variant-specific block kwargs
    block_kwargs = {}

    return ResNetBackbone(
        variant      = variant,
        depth        = depth,
        num_classes  = NUM_CLASSES,
        width_factor = mcfg.get('width_factor', 1.0),
        **block_kwargs,
    )