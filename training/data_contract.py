from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class TrainingExample:
    text: str
    source: str
    split: str = "train"
    quality: float = 1.0
    license: str = "unknown"

    def validate(self) -> "TrainingExample":
        if not self.text.strip():
            raise ValueError("training text cannot be empty")
        if not self.source.strip():
            raise ValueError("training source cannot be empty")
        if self.split not in {"train", "validation", "test"}:
            raise ValueError("invalid dataset split")
        if not 0.0 <= self.quality <= 1.0:
            raise ValueError("quality must be between 0 and 1")
        return self

def validate_batch(examples: list[TrainingExample]) -> list[TrainingExample]:
    if not examples:
        raise ValueError("training batch cannot be empty")
    return [example.validate() for example in examples]
