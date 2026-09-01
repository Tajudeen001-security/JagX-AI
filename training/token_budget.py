from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class TokenBudget:
    max_tokens: int
    used: int = 0

    def __post_init__(self):
        if self.max_tokens <= 0 or self.used < 0 or self.used > self.max_tokens:
            raise ValueError("invalid token budget")

    def consume(self, tokens: int) -> "TokenBudget":
        if tokens < 0:
            raise ValueError("tokens cannot be negative")
        remaining = self.max_tokens - self.used
        if tokens > remaining:
            raise RuntimeError("token budget exceeded")
        return TokenBudget(self.max_tokens, self.used + tokens)

    @property
    def remaining(self) -> int:
        return self.max_tokens - self.used
