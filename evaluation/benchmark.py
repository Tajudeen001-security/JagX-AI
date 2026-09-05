from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass
class EvalResult:
    name: str
    correct: int
    total: int

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0


def load_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def exact_match(prediction: str, answer: str) -> bool:
    return prediction.strip().casefold() == answer.strip().casefold()


def run_benchmark(name: str, rows: list[dict], predict: Callable[[str], str]) -> EvalResult:
    correct = 0
    for row in rows:
        if exact_match(predict(str(row["prompt"])), str(row["answer"])):
            correct += 1
    return EvalResult(name, correct, len(rows))


def save_results(results: list[EvalResult], path: str | Path) -> None:
    Path(path).write_text(json.dumps([
        {"name": r.name, "correct": r.correct, "total": r.total, "accuracy": r.accuracy}
        for r in results
    ], indent=2), encoding="utf-8")
