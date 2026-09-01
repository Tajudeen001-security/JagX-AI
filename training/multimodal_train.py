from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 1000
    grad_accumulation: int = 1
    learning_rate: float = 2e-4
    grad_clip: float = 1.0


def train_steps(model: nn.Module, batches, loss_fn, config: TrainConfig) -> list[float]:
    """Train any native JagX media model from an iterable of prepared batches."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    model.train()
    history: list[float] = []
    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(batches):
        if step >= config.steps:
            break
        loss = loss_fn(model, batch) / config.grad_accumulation
        loss.backward()
        if (step + 1) % config.grad_accumulation == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        history.append(float(loss.detach()) * config.grad_accumulation)
    return history
