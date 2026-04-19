import torch.nn as nn


def build_criterion(cfg: dict) -> nn.Module:
    """Return the loss function specified in cfg['training']['loss'] (default: cross_entropy)."""
    loss_name = cfg.get('training', {}).get('loss', 'cross_entropy')
    if loss_name == 'cross_entropy':
        return nn.CrossEntropyLoss()
    raise ValueError(f"Unknown loss '{loss_name}'")
