from __future__ import annotations
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class CosineSchedule:
    max_lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 100
    total_steps: int = 1000

    def __post_init__(self):
        if self.max_lr <= 0 or self.min_lr < 0 or self.total_steps <= 0 or self.warmup_steps < 0:
            raise ValueError("invalid scheduler configuration")
        if self.warmup_steps >= self.total_steps:
            raise ValueError("warmup_steps must be smaller than total_steps")

    def lr(self, step: int) -> float:
        step = max(0, step)
        if step < self.warmup_steps:
            return self.max_lr * (step + 1) / self.warmup_steps
        progress = min(1.0, (step - self.warmup_steps) / (self.total_steps - self.warmup_steps))
        return self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1.0 + math.cos(math.pi * progress))
