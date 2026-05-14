#!/usr/bin/env python3
"""
scripts/test.py
===============
Evaluation script to test a trained model on CIFAR-10.

Usage examples:
  # Test with checkpoint
  python scripts/test.py --checkpoint runs/exp_name/best.pt --config runs/exp_name/config.json

  # Test with custom config
  python scripts/test.py --checkpoint runs/exp_name/best.pt --config configs/base.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from models import build_model
from training.trainer import Trainer
from training.losses import build_criterion
from utils.data import build_dataloaders
from utils.logging_utils import setup_logging
from utils.checkpoint import CheckpointManager

log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description='Test a trained ResNet model.')
    p.add_argument('--checkpoint', type=str, required=True,
                   help='Path to model checkpoint (.pt file)')
    p.add_argument('--config', type=str, required=True,
                   help='Path to config file (YAML or JSON)')
    p.add_argument('--output_dir', type=str, default=None,
                   help='Optional directory to save results')
    return p.parse_args()


def main():
    args = parse_args()

    # ── Load config ───────────────────────────────────────────────────────
    cfg_path = Path(args.config)
    if cfg_path.suffix == '.json':
        from utils.config import load_json
        cfg = load_json(cfg_path)
    else:
        from utils.config import load_yaml
        cfg = load_yaml(cfg_path)

    # ── Setup logging ─────────────────────────────────────────────────────
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        setup_logging(output_dir)
    else:
        logging.basicConfig(level=logging.INFO)

    log.info(f"Checkpoint: {args.checkpoint}")
    log.info(f"Config: {args.config}")

    # ── Device ────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    log.info(f"Device: {device}")

    # ── Load model ────────────────────────────────────────────────────────
    model = build_model(cfg)
    CheckpointManager.load(Path(args.checkpoint), model, device=str(device))
    log.info(f"Model: {model.variant} depth={model.depth}  "
             f"params={model.num_parameters:,}")

    # ── Load test data ────────────────────────────────────────────────────
    # This loads the official CIFAR-10 test set (10,000 images)
    test_loader = build_dataloaders(cfg, split='test')
    log.info(f"Test set size: {len(test_loader.dataset)}")

    # ── Create criterion and dummy optimizer for trainer ──────────────────
    criterion = build_criterion(cfg)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = None

    # ── Evaluate ──────────────────────────────────────────────────────────
    trainer = Trainer(model, optimizer, scheduler, criterion,
                      device, cfg, Path.cwd())
    model.to(device) 
    model.eval()
    test_metrics = trainer.evaluate(test_loader)

    log.info(f"Test Results - Acc@1: {test_metrics['acc1']:.2f}%  "
             f"Acc@5: {test_metrics.get('acc5', 0):.2f}%  "
             f"Loss: {test_metrics['loss']:.4f}")


if __name__ == '__main__':
    main()
