"""
training/trainer.py
===================
Core training and validation loop.

Features:
  - Automatic mixed precision (AMP) via torch.cuda.amp
  - Per-epoch gradient norm tracking across all named layers
  - Warmup-aware LR scheduling
  - Mixup / CutMix augmentation support
  - Structured metric logging (CSV + TensorBoard)
  - Best-checkpoint and periodic-snapshot saving
"""

import time
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

from evaluation.metrics import AverageMeter, accuracy
from utils.logging_utils import MetricLogger
from utils.checkpoint import CheckpointManager

log = logging.getLogger(__name__)


class Trainer:
    """
    Manages the full training lifecycle for one experiment run.

    Args:
        model       : network to train
        optimizer   : pre-built optimizer
        scheduler   : LR scheduler (stepped per epoch)
        criterion   : loss function
        device      : torch.device
        cfg         : full config dict
        run_dir     : directory for artifacts (checkpoints, logs, plots)
    """

    def __init__(
        self,
        model:     nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        criterion: nn.Module,
        device:    torch.device,
        cfg:       dict,
        run_dir:   Path,
    ):
        self.model     = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device    = device
        self.cfg       = cfg
        self.run_dir   = run_dir

        tcfg = cfg.get('training', {})
        gcfg = cfg.get('gradient_tracking', {})

        self.epochs        = tcfg.get('epochs', 200)
        self.grad_clip     = tcfg.get('grad_clip', None)
        self.use_amp       = tcfg.get('amp', False) and device.type == 'cuda'
        self.mixup_alpha   = tcfg.get('mixup_alpha', 0.0)
        self.log_interval  = tcfg.get('log_interval', 50)

        self.track_grads   = gcfg.get('enabled', True)
        self.grad_freq     = gcfg.get('record_every_n_epochs', 10)

        self.scaler      = GradScaler() if self.use_amp else None
        self.metric_log  = MetricLogger(run_dir, cfg)
        self.ckpt_mgr    = CheckpointManager(run_dir, cfg)

        self.best_acc1   = 0.0
        self.global_step = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> dict:
        """Run training for all epochs. Returns summary metrics."""
        self.model.to(self.device)

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()

            train_metrics = self._train_one_epoch(train_loader, epoch)
            val_metrics   = self._validate(val_loader, epoch)

            if self.scheduler is not None:
                self.scheduler.step()

            epoch_time = time.time() - t0
            lr = self.optimizer.param_groups[0]['lr']

            # Optional gradient norm snapshot
            grad_norms = {}
            if self.track_grads and epoch % self.grad_freq == 0:
                grad_norms = self._collect_grad_norms()

            # Log everything
            combined = {**train_metrics, **val_metrics,
                        'lr': lr, 'epoch_time': epoch_time}
            self.metric_log.log_epoch(epoch, combined, grad_norms)

            # Checkpoint
            is_best = val_metrics['val_acc1'] > self.best_acc1
            if is_best:
                self.best_acc1 = val_metrics['val_acc1']
            self.ckpt_mgr.save(self.model, self.optimizer, epoch,
                               val_metrics, is_best)

            log.info(
                f"Epoch {epoch:3d}/{self.epochs}  "
                f"loss={train_metrics['train_loss']:.4f}  "
                f"val_acc1={val_metrics['val_acc1']:.2f}%  "
                f"best={self.best_acc1:.2f}%  "
                f"lr={lr:.2e}  "
                f"time={epoch_time:.1f}s"
            )

        self.metric_log.close()
        return {'best_val_acc1': self.best_acc1}

    # ------------------------------------------------------------------
    # Private: train one epoch
    # ------------------------------------------------------------------

    def _train_one_epoch(self, loader: DataLoader,
                         epoch: int) -> Dict[str, float]:
        self.model.train()
        loss_m = AverageMeter('train_loss')
        acc1_m = AverageMeter('train_acc1')

        for step, (images, targets) in enumerate(loader, start=1):
            images  = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            if self.mixup_alpha > 0:
                images, targets_a, targets_b, lam = mixup(images, targets,
                                                           self.mixup_alpha)

            with autocast(enabled=self.use_amp):
                logits = self.model(images)
                if self.mixup_alpha > 0:
                    loss = (lam * self.criterion(logits, targets_a)
                            + (1 - lam) * self.criterion(logits, targets_b))
                else:
                    loss = self.criterion(logits, targets)

            self.optimizer.zero_grad(set_to_none=True)

            if self.use_amp:
                self.scaler.scale(loss).backward()
                if self.grad_clip:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(),
                                            self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.grad_clip:
                    nn.utils.clip_grad_norm_(self.model.parameters(),
                                            self.grad_clip)
                self.optimizer.step()

            self.global_step += 1
            acc1 = accuracy(logits.detach(), targets, topk=(1,))[0]
            loss_m.update(loss.item(), images.size(0))
            acc1_m.update(acc1.item(), images.size(0))

            if step % self.log_interval == 0:
                self.metric_log.log_step(self.global_step, {
                    'train_loss': loss_m.avg,
                    'train_acc1': acc1_m.avg,
                    'lr': self.optimizer.param_groups[0]['lr'],
                })

        return {'train_loss': loss_m.avg, 'train_acc1': acc1_m.avg}

    # ------------------------------------------------------------------
    # Private: validation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _validate(self, loader: DataLoader, epoch: int) -> Dict[str, float]:
        self.model.eval()
        loss_m = AverageMeter('val_loss')
        acc1_m = AverageMeter('val_acc1')
        acc5_m = AverageMeter('val_acc5')

        for images, targets in loader:
            images  = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            with autocast(enabled=self.use_amp):
                logits = self.model(images)
                loss   = self.criterion(logits, targets)

            acc1, acc5 = accuracy(logits, targets, topk=(1, 5))
            loss_m.update(loss.item(), images.size(0))
            acc1_m.update(acc1.item(), images.size(0))
            acc5_m.update(acc5.item(), images.size(0))

        return {
            'val_loss': loss_m.avg,
            'val_acc1': acc1_m.avg,
            'val_acc5': acc5_m.avg,
        }

    # ------------------------------------------------------------------
    # Private: gradient norm collection
    # ------------------------------------------------------------------

    def _collect_grad_norms(self) -> Dict[str, float]:
        """
        Return L2 gradient norm for every named parameter that has a gradient.
        Called immediately after the last backward of the epoch.
        """
        norms = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                norms[name] = param.grad.detach().norm(2).item()
        return norms


# ---------------------------------------------------------------------------
# Mixup helper
# ---------------------------------------------------------------------------

def mixup(images: torch.Tensor, targets: torch.Tensor,
          alpha: float) -> Tuple[torch.Tensor, torch.Tensor,
                                  torch.Tensor, float]:
    import numpy as np
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(images.size(0), device=images.device)
    mixed = lam * images + (1 - lam) * images[idx]
    return mixed, targets, targets[idx], lam
