"""
training/optimizers.py
======================
Optimizer and LR scheduler factories with warmup support.
"""

import torch
import torch.nn as nn
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    MultiStepLR,
    OneCycleLR,
    LinearLR,
    SequentialLR,
)

def build_optimizer(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    ocfg = cfg.get('optimizer', {})
    name = ocfg.get('name', 'sgd').lower()
    lr   = ocfg.get('lr', 0.1)
    wd   = ocfg.get('weight_decay', 5e-4)

    if name == 'sgd':
        return SGD(
            model.parameters(), lr=lr, weight_decay=wd,
            momentum=ocfg.get('momentum', 0.9),
            nesterov=ocfg.get('nesterov', True),
        )
    elif name == 'adam':
        return Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif name == 'adamw':
        return AdamW(model.parameters(), lr=lr, weight_decay=wd)
    else:
        raise ValueError(f"Unknown optimizer '{name}'. Choose sgd | adam | adamw.")

#Build an optimizer for the scaled variant — excludes alpha params from weight decay."""
def build_optimizer_scaled(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    ocfg = cfg.get('optimizer', {})
    name = ocfg.get('name', 'sgd').lower()
    lr   = ocfg.get('lr', 0.1)
    wd   = ocfg.get('weight_decay', 5e-4)

    decay, no_decay = [], []
    for pname, param in model.named_parameters():
        if 'alpha' in pname:
            no_decay.append(param)
        else:
            decay.append(param)
    param_groups = [
        {'params': decay,    'weight_decay': wd},
        {'params': no_decay, 'weight_decay': 0.0},
    ]

    if name == 'sgd':
        return SGD(
            param_groups, lr=lr,
            momentum=ocfg.get('momentum', 0.9),
            nesterov=ocfg.get('nesterov', True),
        )
    elif name == 'adam':
        return Adam(param_groups, lr=lr)
    elif name == 'adamw':
        return AdamW(param_groups, lr=lr)
    else:
        raise ValueError(f"Unknown optimizer '{name}'. Choose sgd | adam | adamw.")


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: dict,
                    steps_per_epoch: int = 0):
    """
    Build an LR scheduler with optional linear warmup.

    Supported main schedules: cosine | step | one_cycle
    Warmup wraps the main schedule via SequentialLR.
    """
    scfg    = cfg.get('scheduler', {})
    name    = scfg.get('name', 'cosine').lower()
    epochs  = cfg.get('training', {}).get('epochs', 200)
    warmup  = scfg.get('warmup_epochs', 5)
    main_epochs = epochs - warmup

    if name == 'cosine':
        main = CosineAnnealingLR(optimizer, T_max=main_epochs, eta_min=1e-6)

    elif name == 'step':
        milestones = [m - warmup for m in scfg.get('milestones', [100, 150])]
        main = MultiStepLR(optimizer, milestones=milestones,
                           gamma=scfg.get('gamma', 0.1))

    elif name == 'one_cycle':
        if steps_per_epoch == 0:
            raise ValueError("steps_per_epoch required for one_cycle scheduler")
        # OneCycleLR manages warmup internally; return directly
        return OneCycleLR(
            optimizer,
            max_lr=cfg.get('optimizer', {}).get('lr', 0.1),
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=scfg.get('pct_start', 0.3),
        )
    else:
        raise ValueError(f"Unknown scheduler '{name}'. Choose cosine | step | one_cycle.")

    if warmup > 0:
        warmup_sched = LinearLR(optimizer, start_factor=1e-3,
                                end_factor=1.0, total_iters=warmup)
        return SequentialLR(optimizer,
                            schedulers=[warmup_sched, main],
                            milestones=[warmup])
    return main
