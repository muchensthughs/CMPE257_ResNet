#!/usr/bin/env python3
"""
scripts/train.py
================
Single-run training entry point.

All training configuration should be specified in YAML files.
CLI is limited to: config paths, output paths, and resuming checkpoints.

Usage examples:
  # Train with default config
  python scripts/train.py

  # Train with experiment config (all settings in YAML)
  python scripts/train.py --exp_config configs/experiments/gated_cbam.yaml

  # Resume from checkpoint
  python scripts/train.py --exp_config configs/experiments/gated_cbam.yaml --resume runs/exp_name/best.pt

  # Custom output location
  python scripts/train.py --exp_config configs/experiments/gated_cbam.yaml --output_dir /tmp/test_runs
"""

import argparse
import logging
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from models import build_model
from training.trainer import Trainer
from training.optimizers import build_optimizer, build_optimizer_scaled, build_scheduler
from training.losses import build_criterion
from utils.data import build_dataloaders
from utils.config import load_yaml, deep_merge, \
    make_run_name, save_config, validate_config
from utils.seed import set_seed
from utils.logging_utils import setup_logging
from utils.checkpoint import CheckpointManager

log = logging.getLogger(__name__)

DEFAULT_CFG = Path(__file__).parent.parent / 'configs' / 'base.yaml'


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description='Train a residual variant on CIFAR-10. '
                    'All model/training config should be in YAML files.'
    )

    # Essential config paths
    p.add_argument('--config', type=str, default=str(DEFAULT_CFG),
                   help='Path to base YAML config (default: configs/base.yaml)')
    p.add_argument('--exp_config', type=str, default=None,
                   help='Optional experiment-specific YAML to merge over base')

    # Runtime / output control
    p.add_argument('--output_dir', type=str, default='runs',
                   help='Root directory for run artifacts')
    p.add_argument('--run_name', type=str, default=None,
                   help='Override auto-generated run name (auto-generated from config if not set)')
    p.add_argument('--resume', type=str, default=None,
                   help='Path to checkpoint to resume from')

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # ── Build config ───────────────────────────────────────────────────────
    cfg = load_yaml(args.config)
    if args.exp_config:
        cfg = deep_merge(cfg, load_yaml(args.exp_config))
    validate_config(cfg)

    # ── Create run directory and save config ──────────────────────────────────────────────────────
    run_name = args.run_name or make_run_name(cfg)
    run_dir  = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir / 'config.json')

    # ── Set up logging environment ────────────────────────────────────────────────────────────
    setup_logging(run_dir)
    log.info(f"Run: {run_name}")
    log.info(f"Variant: {cfg['model']['variant']}")
    log.info(f"Config: {run_dir / 'config.json'}")

    # ── Reproducibility ────────────────────────────────────────────────────
    seed = cfg.get('experiment', {}).get('seed', 42)
    set_seed(seed)

    # ── Device ────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    log.info(f"Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────
    train_loader, val_loader = build_dataloaders(cfg)
    log.info(f"Data: train={len(train_loader.dataset)}  "
             f"val={len(val_loader.dataset)}")

    # ── Model ─────────────────────────────────────────────────────────────
    model = build_model(cfg)
    log.info(f"Model: {model.variant} depth={model.depth}  "
             f"params={model.num_parameters:,}")

    # ── Optionally resume ─────────────────────────────────────────────────
    start_epoch = 0
    if args.resume:
        start_epoch, ckpt_metrics = CheckpointManager.load(
            Path(args.resume), model, device=str(device))
        log.info(f"Resumed from epoch {start_epoch}: {ckpt_metrics}")

    # ── Optimizer / scheduler / criterion ─────────────────────────────────
    if cfg['model']['variant'] == 'scaled':
        optimizer = build_optimizer_scaled(model, cfg)
    else:
        optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg,
                                steps_per_epoch=len(train_loader))
    criterion = build_criterion(cfg)

    # ── Train ─────────────────────────────────────────────────────────────
    trainer = Trainer(model, optimizer, scheduler, criterion,
                      device, cfg, run_dir)
    summary = trainer.fit(train_loader, val_loader)

    log.info(f"Training complete. Best val acc@1: {summary['best_val_acc1']:.2f}%")
    log.info(f"Artifacts saved to: {run_dir}")


if __name__ == '__main__':
    main()
