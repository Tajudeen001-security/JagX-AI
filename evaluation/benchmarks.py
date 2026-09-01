from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Any

@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    prompt: str
    expected: Any
    scorer: Callable[[Any, Any], float]

class BenchmarkSuite:
    def __init__(self, cases: list[BenchmarkCase]): self.cases=cases
    def run(self, predict: Callable[[str], Any]):
        results=[]
        for case in self.cases:
            actual=predict(case.prompt)
            results.append({'name':case.name,'score':float(case.scorer(actual,case.expected))})
        return results

    @staticmethod
    def mean_score(results):
        return sum(r['score'] for r in results)/len(results) if results else 0.0
