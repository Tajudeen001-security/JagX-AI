from __future__ import annotations


class GradientAccumulator:
    """Small utility for effective batches larger than device memory permits."""

    def __init__(self, steps: int):
        if steps < 1:
            raise ValueError("accumulation steps must be positive")
        self.steps = steps
        self.micro_steps = 0

    def backward(self, loss, scaler=None):
        scaled_loss = loss / self.steps
        if scaler is None:
            scaled_loss.backward()
        else:
            scaler.scale(scaled_loss).backward()
        self.micro_steps += 1
        return self.ready

    @property
    def ready(self) -> bool:
        return self.micro_steps >= self.steps

    def reset(self):
        self.micro_steps = 0
