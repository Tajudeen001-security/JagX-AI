from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Iterable, Any
import json
import time


@dataclass(frozen=True)
class BenchmarkResult:
    name: str
    score: float
    samples: int
    seconds: float
    passed: bool
    error: str | None = None


class BenchmarkRunner:
    """Run local JagX benchmark adapters without external AI-provider calls."""

    def __init__(self, threshold: float = 0.0):
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        self.threshold = threshold

    def run(self, name: str, samples: Iterable[Any], evaluator: Callable[[Any], float]) -> BenchmarkResult:
        started = time.perf_counter()
        values: list[float] = []
        error: str | None = None
        try:
            for sample in samples:
                score = float(evaluator(sample))
                if not 0.0 <= score <= 1.0:
                    raise ValueError("benchmark scores must be in [0, 1]")
                values.append(score)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        elapsed = time.perf_counter() - started
        score = sum(values) / len(values) if values else 0.0
        return BenchmarkResult(name, score, len(values), elapsed, error is None and score >= self.threshold, error)

    @staticmethod
    def save(results: Iterable[BenchmarkResult], path: str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = [asdict(result) for result in results]
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
