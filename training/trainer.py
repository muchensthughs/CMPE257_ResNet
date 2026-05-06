import time
import logging
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from evaluation.metrics import AverageMeter, accuracy
from utils.logging_utils import MetricLogger
from utils.checkpoint import CheckpointManager

log = logging.getLogger(__name__)


class Trainer:
    """
    Training and validation loop.

    Args:
        model       : network to train
        optimizer   : pre-built optimizer
        scheduler   : LR scheduler (stepped per epoch)
        criterion   : loss function
        device      : torch.device
        cfg         : full config dict
        run_dir     : directory for artifacts (checkpoints, logs)
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

        self.epochs       = tcfg.get('epochs', 200)
        self.log_interval = tcfg.get('log_interval', 50)
        self.track_grads  = gcfg.get('enabled', True)

        self.metric_log = MetricLogger(run_dir, cfg)
        self.ckpt_mgr   = CheckpointManager(run_dir, cfg)

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
            val_metrics   = self._validate(val_loader)

            if self.scheduler is not None:
                self.scheduler.step()

            epoch_time = time.time() - t0
            lr = self.optimizer.param_groups[0]['lr']

            # Collect Conv2d gradient norms every epoch (paper section 6.2.5)
            grad_norms = self._collect_grad_norms() if self.track_grads else {}

            combined = {**train_metrics, **val_metrics, 'lr': lr, 'epoch_time': epoch_time}
            self.metric_log.log_epoch(epoch, combined, grad_norms)

            is_best = val_metrics['val_acc1'] > self.best_acc1
            if is_best:
                self.best_acc1 = val_metrics['val_acc1']
            self.ckpt_mgr.save(self.model, self.optimizer, epoch, val_metrics, is_best)

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

    def _train_one_epoch(self, loader: DataLoader, epoch: int) -> Dict[str, float]:
        self.model.train()
        loss_m = AverageMeter('train_loss')
        acc1_m = AverageMeter('train_acc1')

        for step, (images, targets) in enumerate(loader, start=1):
            images  = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            logits = self.model(images)
            loss   = self.criterion(logits, targets)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
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
    def _validate(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        loss_m = AverageMeter('val_loss')
        acc1_m = AverageMeter('val_acc1')

        for images, targets in loader:
            images  = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            logits = self.model(images)
            loss   = self.criterion(logits, targets)

            acc1 = accuracy(logits, targets, topk=(1,))[0]
            loss_m.update(loss.item(), images.size(0))
            acc1_m.update(acc1.item(), images.size(0))

        return {
            'val_loss': loss_m.avg,
            'val_acc1': acc1_m.avg,
        }
    
    # ------------------------------------------------------------------
    # Public: evaluation (can be used for final testing)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        """
        Public evaluation method. 
        Can be used for validation during training or final testing.
        """
        self.model.eval()
        
        loss_m = AverageMeter('loss')
        acc1_m = AverageMeter('acc1')
        acc5_m = AverageMeter('acc5')

        for images, targets in loader:
            images  = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            logits = self.model(images)
            loss   = self.criterion(logits, targets)

            # Calculate both Top-1 and Top-5 accuracy
            acc1, acc5 = accuracy(logits, targets, topk=(1, 5))
            
            loss_m.update(loss.item(), images.size(0))
            acc1_m.update(acc1.item(), images.size(0))
            acc5_m.update(acc5.item(), images.size(0))

        return {
            'loss': loss_m.avg,
            'acc1': acc1_m.avg,
            'acc5': acc5_m.avg,
        }

    # ------------------------------------------------------------------
    # Private: gradient norm collection (paper section 6.2.5)
    # ------------------------------------------------------------------

    def _collect_grad_norms(self) -> Dict[str, float]:
        """
        L2 gradient norm for each Conv2d layer after the backward pass.
        Records individual layer norms and the mean across all Conv2d layers.
        """
        norms = {}
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d) and module.weight.grad is not None:
                norms[name] = module.weight.grad.detach().norm(2).item()

        if norms:
            norms['mean_conv2d_grad_norm'] = sum(norms.values()) / len(norms)

        return norms
