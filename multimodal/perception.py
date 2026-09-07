"""Perception contracts for camera, screen and video agents."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class Frame:
    source: str
    timestamp: float
    image: bytes
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Perception:
    objects: list[dict[str, Any]] = field(default_factory=list)
    text: list[str] = field(default_factory=list)
    scene: str = ""
    hazards: list[str] = field(default_factory=list)


class VisionBackend(Protocol):
    def perceive(self, frame: Frame) -> Perception: ...


class PerceptionPipeline:
    def __init__(self, backend: VisionBackend):
        self.backend = backend

    def process(self, frame: Frame) -> Perception:
        result = self.backend.perceive(frame)
        if not isinstance(result, Perception):
            raise TypeError("vision backend must return Perception")
        return result
