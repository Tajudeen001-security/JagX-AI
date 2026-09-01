from __future__ import annotations
import random
import os


def seed_everything(seed: int = 42) -> None:
    """Make supported training components deterministic where practical."""
    if seed < 0:
        raise ValueError('seed must be non-negative')
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
