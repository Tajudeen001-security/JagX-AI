from __future__ import annotations
from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class Checkpoint:
    step: int
    path: str
    loss: float
    eval_score: float


class CheckpointManifest:
    def __init__(self, path: str = "checkpoints/manifest.json"):
        self.path = Path(path)
        self.items: list[Checkpoint] = []

    def add(self, checkpoint: Checkpoint):
        if checkpoint.step < 0 or checkpoint.loss < 0:
            raise ValueError("invalid checkpoint metrics")
        self.items.append(checkpoint)
        self.items.sort(key=lambda item: item.step)

    def best(self) -> Checkpoint | None:
        return max(self.items, key=lambda item: item.eval_score, default=None)

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps([asdict(item) for item in self.items], indent=2), encoding="utf-8")
        tmp.replace(self.path)
