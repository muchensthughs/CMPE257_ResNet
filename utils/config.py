"""
utils/config.py
===============
Load, deep-merge, validate, and serialise experiment configurations.
"""

import copy
import json
import yaml
from pathlib import Path
from typing import Any, Dict, Optional


def load_yaml(path: str | Path) -> Dict[str, Any]:
    """Load a YAML file into a plain dict."""
    with open(path) as f:
        return yaml.safe_load(f) or {}


def deep_merge(base: dict, override: dict) -> dict:
    """
    Recursively merge `override` into `base`.

    Nested dicts are merged; all other types are replaced.
    Neither argument is mutated.
    """
    result = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = copy.deepcopy(v)
    return result


def apply_cli_overrides(cfg: dict, overrides: Dict[str, Any]) -> dict:
    """
    Apply flat dot-notation overrides from CLI arguments.

    Example:
        overrides = {'model.variant': 'gated', 'training.epochs': 100}
    """
    cfg = copy.deepcopy(cfg)
    for dotkey, value in overrides.items():
        parts = dotkey.split('.')
        node  = cfg
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        # Type-cast strings for common types
        node[parts[-1]] = _cast(value)
    return cfg


def _cast(value: Any) -> Any:
    """Try to cast a string CLI value to int, float, bool, or None."""
    if not isinstance(value, str):
        return value
    if value.lower() == 'none':
        return None
    if value.lower() == 'true':
        return True
    if value.lower() == 'false':
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def make_run_name(cfg: dict) -> str:
    """
    Generate a descriptive run name from config, e.g.:
        'depth20_gated_cbam'
    """
    mcfg    = cfg.get('model', {})
    parts   = [
        f"d{mcfg.get('depth', '?')}",
        mcfg.get('variant', 'unknown'),
    ]
    if mcfg.get('attention'):
        parts.append(mcfg['attention'])
    if mcfg.get('variant') == 'scaled' and not mcfg.get('learnable_alpha', True):
        parts.append(f"a{mcfg.get('alpha', 1.0)}")

    extra = cfg.get('experiment', {}).get('tags', [])
    parts.extend(extra)
    return '_'.join(str(p) for p in parts)


def save_config(cfg: dict, path: Path):
    """Save config as JSON for reproducibility."""
    with open(path, 'w') as f:
        json.dump(cfg, f, indent=2, default=str)


def validate_config(cfg: dict):
    """
    Raise ValueError if required keys are missing or invalid.
    """
    required = ['model', 'training']
    for key in required:
        if key not in cfg:
            raise ValueError(f"Config missing required top-level key: '{key}'")

    valid_variants = {'baseline', 'no_residual', 'scaled', 'gated'}
    variant = cfg['model'].get('variant')
    if variant not in valid_variants:
        raise ValueError(
            f"model.variant='{variant}' is invalid. "
            f"Choose from {sorted(valid_variants)}"
        )
