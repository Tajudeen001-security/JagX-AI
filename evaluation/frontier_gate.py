from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkResult:
    name: str
    score: float
    baseline: float

    @property
    def exceeds_baseline(self) -> bool:
        return self.score > self.baseline


@dataclass(frozen=True)
class CapabilityGate:
    required: tuple[str, ...]

    def evaluate(self, results: list[BenchmarkResult]) -> dict[str, bool]:
        by_name = {result.name: result.exceeds_baseline for result in results}
        return {name: by_name.get(name, False) for name in self.required}

    def frontier_claim_allowed(self, results: list[BenchmarkResult]) -> bool:
        gate = self.evaluate(results)
        return bool(gate) and all(gate.values())
