from __future__ import annotations
import random
from typing import TypeVar

T = TypeVar("T")

def deterministic_shuffle(items: list[T], seed: int) -> list[T]:
    if seed < 0:
        raise ValueError("seed must be non-negative")
    result = list(items)
    random.Random(seed).shuffle(result)
    return result

def shard(items: list[T], rank: int, world_size: int) -> list[T]:
    if world_size < 1 or rank < 0 or rank >= world_size:
        raise ValueError("invalid distributed shard parameters")
    return items[rank::world_size]
