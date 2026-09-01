from __future__ import annotations
from dataclasses import dataclass
import re

@dataclass(frozen=True)
class QualityPolicy:
    min_chars: int = 20
    max_chars: int = 200_000
    min_alpha_ratio: float = 0.20
    reject_repeated_line_ratio: float = 0.50

def score_text(text: str, policy: QualityPolicy = QualityPolicy()) -> float:
    if not text or len(text) < policy.min_chars or len(text) > policy.max_chars:
        return 0.0
    chars = sum(c.isalpha() for c in text)
    alpha_ratio = chars / max(1, len(text))
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    repeated = 0.0
    if lines:
        repeated = 1.0 - len(set(lines)) / len(lines)
    score = min(1.0, alpha_ratio / max(policy.min_alpha_ratio, 1e-9))
    if repeated >= policy.reject_repeated_line_ratio:
        score *= 0.25
    return max(0.0, min(1.0, score))

def accept(text: str, policy: QualityPolicy = QualityPolicy()) -> bool:
    return score_text(text, policy) >= 0.5
