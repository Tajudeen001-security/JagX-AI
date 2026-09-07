"""Small, model-independent learning store for JagX task experience.

This is not weight training. It records outcomes so a future model can retrieve
successful procedures without retraining the base model after every task.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class TaskExperience:
    goal: str
    steps: list[str]
    outcome: str
    success: bool
    tags: list[str]
    created_at: float


class ExperienceStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def add(self, goal: str, steps: list[str], outcome: str, success: bool, tags: list[str] | None = None) -> None:
        item = TaskExperience(goal, steps, outcome, success, tags or [], time.time())
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")

    def search(self, query: str, limit: int = 5) -> list[TaskExperience]:
        if not self.path.exists():
            return []
        terms = {x.lower() for x in query.split() if len(x) > 2}
        rows: list[tuple[int, TaskExperience]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                raw = json.loads(line)
                item = TaskExperience(**raw)
            except (ValueError, TypeError, json.JSONDecodeError):
                continue
            haystack = f"{item.goal} {item.outcome} {' '.join(item.tags)}".lower()
            score = sum(term in haystack for term in terms)
            if score:
                rows.append((score, item))
        rows.sort(key=lambda pair: (pair[0], pair[1].created_at), reverse=True)
        return [item for _, item in rows[:limit]]
