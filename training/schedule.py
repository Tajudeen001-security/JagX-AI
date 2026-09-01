from __future__ import annotations

import math


def cosine_lr(step: int, total_steps: int, warmup_steps: int, max_lr: float, min_lr: float = 0.0) -> float:
    if step < warmup_steps:
        return max_lr * (step + 1) / max(1, warmup_steps)
    progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))
