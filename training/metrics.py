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
            raise ValueError("loss must be finite and non-negative")
        if tokens < 0:
            raise ValueError("tokens cannot be negative")
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
        return {
            "steps": self.steps,
            "mean_loss": self.mean_loss,
            "perplexity": self.perplexity,
            "tokens": self.total_tokens,
        }

    @classmethod
    def from_snapshot(cls, snapshot: dict | None) -> "RunningMetrics":
        """Reconstruct aggregate metrics saved in a checkpoint."""
        data = snapshot or {}
        steps = int(data.get("steps", 0))
        tokens = int(data.get("tokens", 0))
        mean_loss = float(data.get("mean_loss", 0.0))
        if steps < 0 or tokens < 0 or not math.isfinite(mean_loss) or mean_loss < 0:
            raise ValueError("invalid metrics snapshot")
        return cls(steps=steps, total_loss=mean_loss * steps, total_tokens=tokens)
