from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class CameraFrame:
    timestamp_ms: int
    camera_id: str
    width: int
    height: int
    encoding: str
    payload: bytes = field(repr=False)


@dataclass(frozen=True)
class ObjectObservation:
    label: str
    confidence: float
    distance_m: float | None = None
    relative_speed_mps: float | None = None
    bbox: tuple[float, float, float, float] | None = None


class VisionModel(Protocol):
    def detect(self, frame: CameraFrame) -> list[ObjectObservation]: ...


class VehiclePerception:
    """Camera/sensor abstraction. Host applications provide the actual camera feed."""

    def __init__(self, vision_model: VisionModel):
        self.vision_model = vision_model

    def process(self, frame: CameraFrame) -> dict[str, Any]:
        observations = self.vision_model.detect(frame)
        return {
            "camera_id": frame.camera_id,
            "timestamp_ms": frame.timestamp_ms,
            "objects": [o.__dict__ for o in observations],
        }
