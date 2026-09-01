from __future__ import annotations
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class RegressionCase:
    name: str
    prompt: str
    minimum_score: float


class RegressionSuite:
    def __init__(self, cases: list[RegressionCase]):
        self.cases = cases

    def run(self, predict: Callable[[str], float]):
        failures = []
        for case in self.cases:
            score = float(predict(case.prompt))
            if score < case.minimum_score:
                failures.append({"name": case.name, "score": score, "minimum": case.minimum_score})
        return failures
