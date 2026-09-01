from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class MovieSpec:
    title: str
    duration_seconds: int = 1200
    fps: int = 24
    width: int = 1920
    height: int = 1080

    def validate(self):
        if not 1 <= self.duration_seconds <= 1200:
            raise ValueError("duration must be 1..1200 seconds")
        if self.fps not in (24, 25, 30, 48, 60):
            raise ValueError("unsupported fps")
        if self.width < 256 or self.height < 256:
            raise ValueError("resolution too small")


class MoviePipeline:
    """Orchestrates long-form generation as independently verifiable shots."""

    def __init__(self, scene_generator, frame_generator, video_generator):
        self.scene_generator = scene_generator
        self.frame_generator = frame_generator
        self.video_generator = video_generator

    def plan(self, spec: MovieSpec):
        spec.validate()
        return self.scene_generator.create_plan(spec)

    def render_shot(self, shot, spec: MovieSpec):
        keyframes = self.frame_generator.generate(shot, spec)
        return self.video_generator.generate(shot, keyframes, spec)
