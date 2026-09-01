from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelScale:
    name: str
    layers: int
    d_model: int
    heads: int
    context: int
    vocab: int = 32768


SCALES = {
    "tiny": ModelScale("tiny", 6, 384, 6, 2048),
    "small": ModelScale("small", 12, 768, 12, 4096),
    "medium": ModelScale("medium", 24, 1536, 24, 8192),
    "large": ModelScale("large", 32, 4096, 32, 16384),
}


def get_scale(name: str) -> ModelScale:
    if name not in SCALES:
        raise ValueError(f"unknown scale: {name}")
    return SCALES[name]
