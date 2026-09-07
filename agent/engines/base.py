"""Engine adapter contract for agentic project creation."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class ProjectRequest:
    project_dir: Path
    task: str
    language: str = "en"
    platform: str | None = None


@dataclass
class EngineResult:
    success: bool
    command: Sequence[str] = field(default_factory=tuple)
    stdout: str = ""
    stderr: str = ""
    artifacts: list[str] = field(default_factory=list)


class EngineAdapter:
    name = "base"

    def available(self) -> bool:
        return False

    def scaffold(self, request: ProjectRequest) -> EngineResult:
        raise NotImplementedError

    def run(self, request: ProjectRequest) -> EngineResult:
        raise NotImplementedError

    def test(self, request: ProjectRequest) -> EngineResult:
        raise NotImplementedError

    def export(self, request: ProjectRequest, preset: str) -> EngineResult:
        raise NotImplementedError
