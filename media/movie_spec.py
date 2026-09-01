from dataclasses import dataclass

MAX_DURATION_SECONDS = 20 * 60


@dataclass(frozen=True)
class MovieSpec:
    title: str
    duration_seconds: int
    fps: int = 24
    width: int = 1920
    height: int = 1080

    def validate(self):
        if not 1 <= self.duration_seconds <= MAX_DURATION_SECONDS:
            raise ValueError(f"duration must be between 1 and {MAX_DURATION_SECONDS} seconds")
        if self.fps not in (24, 25, 30, 48, 50, 60):
            raise ValueError("unsupported fps")
        if self.width < 256 or self.height < 256:
            raise ValueError("resolution too small")
