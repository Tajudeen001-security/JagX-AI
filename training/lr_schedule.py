from __future__ import annotations
import math


def cosine_lr(step: int, total_steps: int, warmup_steps: int, max_lr: float, min_lr: float = 0.0) -> float:
    if total_steps <= 0 or warmup_steps < 0 or warmup_steps >= total_steps:
        raise ValueError('invalid step configuration')
    if max_lr <= 0 or min_lr < 0 or min_lr > max_lr:
        raise ValueError('invalid learning rates')
    if step < 0:
        raise ValueError('step cannot be negative')
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    progress = min(1.0, (step - warmup_steps) / (total_steps - warmup_steps))
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))
