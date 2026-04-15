"""
utils/seed.py
=============
Reproducibility utilities.
"""

import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    """Set all RNG seeds for reproducible results."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
