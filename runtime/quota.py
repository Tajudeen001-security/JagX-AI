from __future__ import annotations
from dataclasses import dataclass
from time import monotonic


@dataclass
class TokenBucket:
    capacity: int
    refill_per_second: float
    tokens: float | None = None
    updated: float | None = None

    def __post_init__(self):
        if self.capacity <= 0 or self.refill_per_second <= 0:
            raise ValueError("invalid bucket limits")
        now = monotonic()
        self.tokens = float(self.capacity) if self.tokens is None else min(float(self.tokens), float(self.capacity))
        self.updated = now if self.updated is None else now

    def consume(self, cost: int = 1) -> bool:
        if cost < 0:
            raise ValueError("cost cannot be negative")
        now = monotonic()
        elapsed = now - self.updated
        self.tokens = min(float(self.capacity), self.tokens + elapsed * self.refill_per_second)
        self.updated = now
        if self.tokens < cost:
            return False
        self.tokens -= cost
        return True
