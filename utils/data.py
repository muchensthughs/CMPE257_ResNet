"""
utils/data.py
=============
Dataset and DataLoader factory for CIFAR-10.

Augmentation options (all toggled via config):
  - Standard: RandomCrop + HorizontalFlip
  - AutoAugment
  - Random Erasing
"""

from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as T


# ---------------------------------------------------------------------------
# CIFAR-10 normalization constants
# ---------------------------------------------------------------------------

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


# ---------------------------------------------------------------------------
# Transform builders
# ---------------------------------------------------------------------------

def _build_train_transforms(cfg: dict) -> T.Compose:
    """Build training transforms with optional augmentations."""
    dcfg = cfg.get('augmentation', {})
    tfms = []

    if dcfg.get('random_crop', True):
        tfms.append(T.RandomCrop(32, padding=4))
    if dcfg.get('random_flip', True):
        tfms.append(T.RandomHorizontalFlip())
    if dcfg.get('autoaugment', False):
        tfms.append(T.AutoAugment(T.AutoAugmentPolicy.CIFAR10))
    if dcfg.get('trivial_augment', False):
        tfms.append(T.TrivialAugmentWide())

    tfms.append(T.ToTensor())
    tfms.append(T.Normalize(CIFAR10_MEAN, CIFAR10_STD))

    if dcfg.get('random_erasing', False):
        tfms.append(T.RandomErasing(p=0.5))

    return T.Compose(tfms)


def _build_val_transforms() -> T.Compose:
    """Build validation transforms (minimal, no augmentation)."""
    return T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


# ---------------------------------------------------------------------------
# Main factory
# ---------------------------------------------------------------------------

def build_dataloaders(cfg: dict) -> Tuple[DataLoader, DataLoader]:
    """
    Build (train_loader, val_loader) for CIFAR-10.

    Config keys used:
      training.batch_size
      training.num_workers
      training.pin_memory
    """
    tcfg    = cfg.get('training', {})
    root    = Path('./data')
    bs      = tcfg.get('batch_size', 128)
    workers = tcfg.get('num_workers', 4)
    pin     = tcfg.get('pin_memory', True)

    train_ds = datasets.CIFAR10(
        root, train=True, download=True,
        transform=_build_train_transforms(cfg))
    val_ds = datasets.CIFAR10(
        root, train=False, download=True,
        transform=_build_val_transforms())

    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True,
        num_workers=workers, pin_memory=pin,
        drop_last=True, persistent_workers=(workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs * 2, shuffle=False,
        num_workers=workers, pin_memory=pin,
        persistent_workers=(workers > 0),
    )

    return train_loader, val_loader
