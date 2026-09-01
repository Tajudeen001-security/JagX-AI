from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class OptimizerConfig:
    learning_rate: float = 3e-4
    betas: tuple[float,float] = (0.9, 0.95)
    eps: float = 1e-8
    weight_decay: float = 0.1
    grad_clip: float = 1.0

    def validate(self):
        if self.learning_rate <= 0 or self.eps <= 0 or self.grad_clip <= 0:
            raise ValueError('optimizer values must be positive')
        if not 0 < self.betas[0] < 1 or not 0 < self.betas[1] < 1:
            raise ValueError('optimizer betas must be between 0 and 1')
        if self.weight_decay < 0:
            raise ValueError('weight decay cannot be negative')
        return self

    def build(self, parameters):
        import torch
        self.validate()
        return torch.optim.AdamW(parameters, lr=self.learning_rate, betas=self.betas, eps=self.eps, weight_decay=self.weight_decay)
