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
import numpy as np
from torch.utils.data import DataLoader, Subset, Dataset
import torchvision.datasets as datasets
import torchvision.transforms as T


# ---------------------------------------------------------------------------
# CIFAR-10 normalization constants
# ---------------------------------------------------------------------------

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

# ---------------------------------------------------------------------------
# Transform Wrapper
# ---------------------------------------------------------------------------

class ApplyTransform(Dataset):
    """
    A small wrapper to apply specific transforms to a Subset of a dataset.
    """
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
        
    def __len__(self):
        return len(self.subset)

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

def build_dataloaders(cfg: dict, split: str = 'train_val') -> Tuple[DataLoader, DataLoader]:
    """
    Build dataloaders for CIFAR-10.

    Args:
        cfg: Configuration dictionary
        split: One of 'train_val' (returns train+val), 'test' (returns test only), or 'all' (returns all three)

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
    val_split = tcfg.get('val_split', 0.1)

    if split in ['train_val', 'all']:
        # 1. Load the official Training set (50,000 images)
        # We load without transforms initially to apply them via our wrapper
        full_train_ds = datasets.CIFAR10(root, train=True, download=True)
        
        # 2. Create deterministic indices for the split
        indices = list(range(len(full_train_ds)))
        split_idx = int(np.floor(val_split * len(full_train_ds)))
        
        # Using a fixed seed ensures your Val set is the same every time you run
        np.random.seed(42)
        np.random.shuffle(indices)
        
        train_idx, val_idx = indices[split_idx:], indices[:split_idx]

        # 3. Create Subsets with specific transforms
        train_ds = ApplyTransform(
            Subset(full_train_ds, train_idx), 
            transform=_build_train_transforms(cfg)
        )
        val_ds = ApplyTransform(
            Subset(full_train_ds, val_idx), 
            transform=_build_val_transforms()
        )

        train_loader = DataLoader(
            train_ds, batch_size=bs, shuffle=True,
            num_workers=workers, pin_memory=pin, drop_last=True
        )
        val_loader = DataLoader(
            val_ds, batch_size=bs * 2, shuffle=False,
            num_workers=workers, pin_memory=pin
        )

        if split == 'train_val':
            return train_loader, val_loader

    # 4. Handle Test Split (Official 10,000 images)
    if split in ['test', 'all']:
        # using train=False ensures we are consistently 
        # loading the CIFAR-10 test set (10,000 images), 
        # which is a static, pre-defined subset of the original data
        test_ds = datasets.CIFAR10(
            root, train=False, download=True,
            transform=_build_val_transforms()
        )
        test_loader = DataLoader(
            test_ds, batch_size=bs * 2, shuffle=False,
            num_workers=workers, pin_memory=pin
        )
        
        if split == 'test':
            return test_loader
        else: # split == 'all'
            return train_loader, val_loader, test_loader
