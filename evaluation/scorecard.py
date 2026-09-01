from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class Score:
    name: str
    value: float
    higher_is_better: bool = True


class ReleaseScorecard:
    def __init__(self, scores: list[Score]):
        if not scores:
            raise ValueError("at least one score is required")
        self.scores = tuple(scores)

    def summary(self) -> dict[str, float]:
        return {s.name: s.value for s in self.scores}

    def passes(self, minimums: dict[str, float]) -> bool:
        for score in self.scores:
            if score.name not in minimums:
                continue
            threshold = minimums[score.name]
            if score.higher_is_better and score.value < threshold:
                return False
            if not score.higher_is_better and score.value > threshold:
                return False
        return True
