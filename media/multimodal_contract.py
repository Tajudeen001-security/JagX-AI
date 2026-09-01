from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class Modality(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"


@dataclass(frozen=True)
class MediaTensor:
    modality: Modality
    values: Any
    sample_rate: int | None = None
    frame_rate: float | None = None
    metadata: dict[str, Any] | None = None

    def validate(self) -> "MediaTensor":
        if self.values is None:
            raise ValueError("media tensor values cannot be None")
        if self.modality is Modality.AUDIO and self.sample_rate is not None and self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.modality is Modality.VIDEO and self.frame_rate is not None and self.frame_rate <= 0:
            raise ValueError("frame_rate must be positive")
        return self


@dataclass(frozen=True)
class MultimodalBatch:
    items: tuple[MediaTensor, ...]

    def validate(self) -> "MultimodalBatch":
        for item in self.items:
            item.validate()
        return self

    def modalities(self) -> set[Modality]:
        return {item.modality for item in self.items}
