"""
utils/checkpoint.py
===================
Save and load model checkpoints.
"""

from pathlib import Path
from typing import Optional, Tuple
import torch
import torch.nn as nn


class CheckpointManager:
    """
    Saves checkpoints to run_dir:
      best.pt            — highest val_acc1 so far
      last.pt            — most recent epoch
      epoch_{N:04d}.pt   — periodic snapshots

    Args:
        run_dir  : directory to write checkpoints
        cfg      : config dict  (reads logging.save_every_n_epochs)
    """

    def __init__(self, run_dir: Path, cfg: dict):
        self.run_dir    = run_dir
        self.save_every = cfg.get('logging', {}).get('save_every_n_epochs', 0)
        run_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        model:     nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch:     int,
        metrics:   dict,
        is_best:   bool,
    ):
        state = {
            'epoch':     epoch,
            'model':     model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'metrics':   metrics,
        }

        torch.save(state, self.run_dir / 'last.pt')

        if is_best:
            torch.save(state, self.run_dir / 'best.pt')

        if self.save_every and epoch % self.save_every == 0:
            torch.save(state, self.run_dir / f'epoch_{epoch:04d}.pt')

    @staticmethod
    def load(
        path:      Path,
        model:     nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device:    str = 'cpu',
    ) -> Tuple[int, dict]:
        """Load a checkpoint. Returns (epoch, metrics)."""
        ckpt = torch.load(path, map_location=device)
        model.load_state_dict(ckpt['model'])
        if optimizer is not None and 'optimizer' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer'])
        return ckpt.get('epoch', 0), ckpt.get('metrics', {})
