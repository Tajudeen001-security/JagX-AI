from __future__ import annotations
from dataclasses import dataclass

from .data_contract import TrainingExample, validate_batch
from .dedup import ExactDeduplicator
from .quality_filter import QualityPolicy, accept
from .shuffle import deterministic_shuffle, shard


@dataclass(frozen=True)
class PipelineStats:
    received: int
    accepted: int
    rejected_quality: int
    rejected_duplicate: int


class CorpusPipeline:
    def __init__(self, quality_policy: QualityPolicy = QualityPolicy()):
        self.quality_policy = quality_policy
        self.dedup = ExactDeduplicator()

    def process(
        self, examples: list[TrainingExample], seed: int = 42, rank: int = 0, world_size: int = 1
    ) -> tuple[list[TrainingExample], PipelineStats]:
        validate_batch(examples)
        ordered = deterministic_shuffle(examples, seed)
        local = shard(ordered, rank, world_size)
        output: list[TrainingExample] = []
        quality_rejected = duplicate_rejected = 0

        for example in local:
            if not accept(example.text, self.quality_policy):
                quality_rejected += 1
                continue
            if not self.dedup.accept(example.text):
                duplicate_rejected += 1
                continue
            output.append(example)

        return output, PipelineStats(len(local), len(output), quality_rejected, duplicate_rejected)
