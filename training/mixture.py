from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetSource:
    name: str
    weight: float


class DatasetMixture:
    """Deterministic normalized weights for composing training sources."""

    def __init__(self, sources: list[DatasetSource]):
        if not sources:
            raise ValueError("at least one dataset source is required")
        if any(s.weight < 0 for s in sources):
            raise ValueError("dataset weights cannot be negative")
        total = sum(s.weight for s in sources)
        if total <= 0:
            raise ValueError("dataset weights must have positive total")
        self.sources = tuple(DatasetSource(s.name, s.weight / total) for s in sources)

    def weights(self) -> dict[str, float]:
        return {s.name: s.weight for s in self.sources}
