from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MovieSpec:
    title: str
    duration_seconds: int = 1200
    fps: int = 24
    width: int = 1920
    height: int = 1080
    shot_seconds: int = 10

    def validate(self) -> None:
        if not 1 <= self.duration_seconds <= 86_400:
            raise ValueError("duration must be 1..86400 seconds")
        if self.fps not in (24, 25, 30, 48, 60):
            raise ValueError("unsupported fps")
        if self.width < 256 or self.height < 256:
            raise ValueError("resolution too small")
        if self.shot_seconds < 1:
            raise ValueError("shot_seconds must be positive")


@dataclass(frozen=True)
class Shot:
    index: int
    start_seconds: int
    duration_seconds: int


class MoviePipeline:
    """Long-form generation orchestrator with resumable shot boundaries."""

    def __init__(self, scene_generator, frame_generator, video_generator):
        self.scene_generator = scene_generator
        self.frame_generator = frame_generator
        self.video_generator = video_generator

    def plan(self, spec: MovieSpec):
        spec.validate()
        return self.scene_generator.create_plan(spec)

    def shots(self, spec: MovieSpec) -> list[Shot]:
        spec.validate()
        return [
            Shot(i, start, min(spec.shot_seconds, spec.duration_seconds - start))
            for i, start in enumerate(range(0, spec.duration_seconds, spec.shot_seconds))
        ]

    def render_shot(self, shot, spec: MovieSpec):
        keyframes = self.frame_generator.generate(shot, spec)
        return self.video_generator.generate(shot, keyframes, spec)

    def render(self, spec: MovieSpec, *, completed: set[int] | None = None) -> dict[int, object]:
        done = completed or set()
        outputs = {}
        for shot in self.shots(spec):
            if shot.index in done:
                continue
            outputs[shot.index] = self.render_shot(shot, spec)
        return outputs
