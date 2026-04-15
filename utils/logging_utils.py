"""
utils/logging_utils.py
======================
Structured metric logging to CSV and optionally TensorBoard.

MetricLogger writes:
  metrics_epoch.csv  — one row per epoch (train/val metrics, LR)
  metrics_step.csv   — one row per log_interval steps
  gradient_norms.csv — per-layer gradient norm snapshots
"""

import csv
import logging
from pathlib import Path
from typing import Dict, Any, Optional

log = logging.getLogger(__name__)


def setup_logging(run_dir: Path, level: int = logging.INFO):
    """Configure Python logging to file + stdout for a run."""
    run_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s',
                            datefmt='%H:%M:%S')
    handlers = [
        logging.StreamHandler(),
        logging.FileHandler(run_dir / 'train.log'),
    ]
    root = logging.getLogger()
    root.setLevel(level)
    for h in handlers:
        h.setFormatter(fmt)
        root.addHandler(h)


class MetricLogger:
    """
    Logs epoch and step metrics to CSV and optionally TensorBoard.

    Args:
        run_dir : directory for output files
        cfg     : config dict (reads logging.tensorboard flag)
    """

    def __init__(self, run_dir: Path, cfg: dict):
        run_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir = run_dir

        lcfg = cfg.get('logging', {})
        self.use_tb = lcfg.get('tensorboard', True)

        self._epoch_file:   Optional[Any] = None
        self._epoch_writer: Optional[csv.DictWriter] = None
        self._step_file:    Optional[Any] = None
        self._step_writer:  Optional[csv.DictWriter] = None
        self._grad_file:    Optional[Any] = None
        self._grad_writer:  Optional[csv.DictWriter] = None
        self._tb:           Optional[Any] = None

        if self.use_tb:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._tb = SummaryWriter(log_dir=str(run_dir / 'tb'))
            except ImportError:
                log.warning("TensorBoard not installed; skipping TB logging.")
                self.use_tb = False

    # ------------------------------------------------------------------

    def log_epoch(self, epoch: int, metrics: Dict[str, float],
                  grad_norms: Optional[Dict[str, float]] = None):
        row = {'epoch': epoch, **metrics}

        # CSV
        if self._epoch_writer is None:
            self._epoch_file = open(self.run_dir / 'metrics_epoch.csv',
                                    'w', newline='')
            self._epoch_writer = csv.DictWriter(
                self._epoch_file, fieldnames=list(row.keys()))
            self._epoch_writer.writeheader()
        self._epoch_writer.writerow(row)
        self._epoch_file.flush()

        # TensorBoard
        if self._tb:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self._tb.add_scalar(f'epoch/{k}', v, epoch)

        # Gradient norms
        if grad_norms:
            self.log_grad_norms(epoch, grad_norms)

    def log_step(self, step: int, metrics: Dict[str, float]):
        row = {'step': step, **metrics}

        if self._step_writer is None:
            self._step_file = open(self.run_dir / 'metrics_step.csv',
                                   'w', newline='')
            self._step_writer = csv.DictWriter(
                self._step_file, fieldnames=list(row.keys()))
            self._step_writer.writeheader()
        self._step_writer.writerow(row)
        self._step_file.flush()

        if self._tb:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self._tb.add_scalar(f'step/{k}', v, step)

    def log_grad_norms(self, epoch: int, norms: Dict[str, float]):
        if self._grad_writer is None:
            self._grad_file = open(self.run_dir / 'gradient_norms.csv',
                                   'w', newline='')
            self._grad_writer = csv.DictWriter(
                self._grad_file,
                fieldnames=['epoch', 'layer', 'grad_norm'])
            self._grad_writer.writeheader()

        for layer, norm in norms.items():
            self._grad_writer.writerow({
                'epoch': epoch, 'layer': layer, 'grad_norm': norm})
        self._grad_file.flush()

        if self._tb:
            for layer, norm in norms.items():
                safe = layer.replace('.', '/')
                self._tb.add_scalar(f'grad_norm/{safe}', norm, epoch)

    def close(self):
        for f in (self._epoch_file, self._step_file, self._grad_file):
            if f:
                f.close()
        if self._tb:
            self._tb.close()
