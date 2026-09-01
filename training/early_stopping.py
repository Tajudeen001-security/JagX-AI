from __future__ import annotations


class EarlyStopping:
    """Stop a run when evaluation quality stops improving."""

    def __init__(self, patience: int = 5, min_delta: float = 0.0):
        if patience < 1 or min_delta < 0:
            raise ValueError("patience must be positive and min_delta non-negative")
        self.patience = patience
        self.min_delta = min_delta
        self.best: float | None = None
        self.bad_steps = 0

    def update(self, score: float) -> bool:
        if not isinstance(score, (int, float)) or score != score:
            raise ValueError("score must be finite")
        if self.best is None or score > self.best + self.min_delta:
            self.best = float(score)
            self.bad_steps = 0
            return False
        self.bad_steps += 1
        return self.bad_steps >= self.patience

    def reset(self) -> None:
        self.best = None
        self.bad_steps = 0
