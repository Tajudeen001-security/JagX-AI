from __future__ import annotations
from dataclasses import dataclass
import math

@dataclass
class RunningMetrics:
    steps: int = 0
    total_loss: float = 0.0
    total_tokens: int = 0

    def update(self, loss: float, tokens: int = 0) -> None:
        if not math.isfinite(loss) or loss < 0:
            raise ValueError('loss must be finite and non-negative')
        if tokens < 0:
            raise ValueError('tokens cannot be negative')
        self.steps += 1
        self.total_loss += loss
        self.total_tokens += tokens

    @property
    def mean_loss(self) -> float:
        return self.total_loss / self.steps if self.steps else 0.0

    @property
    def perplexity(self) -> float:
        return math.exp(min(self.mean_loss, 20.0)) if self.steps else 0.0

    def snapshot(self) -> dict:
        return {'steps': self.steps, 'mean_loss': self.mean_loss, 'perplexity': self.perplexity, 'tokens': self.total_tokens}
