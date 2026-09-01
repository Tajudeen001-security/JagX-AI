from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetStats:
    examples: int
    characters: int
    estimated_tokens: int

    @property
    def chars_per_token(self) -> float:
        return self.characters / self.estimated_tokens if self.estimated_tokens else 0.0


def estimate_tokens(text: str, chars_per_token: float = 4.0) -> int:
    if chars_per_token <= 0:
        raise ValueError("chars_per_token must be positive")
    return max(1, int(len(text) / chars_per_token)) if text else 0


def summarize(texts: list[str], chars_per_token: float = 4.0) -> DatasetStats:
    if not texts:
        raise ValueError("dataset cannot be empty")
    characters = sum(len(text) for text in texts)
    tokens = sum(estimate_tokens(text, chars_per_token) for text in texts)
    return DatasetStats(len(texts), characters, tokens)
