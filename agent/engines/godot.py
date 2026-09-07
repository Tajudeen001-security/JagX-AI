"""Godot adapter using the engine's documented CLI workflow."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .base import EngineAdapter, EngineResult, ProjectRequest


class GodotAdapter(EngineAdapter):
    name = "godot"

    def __init__(self, executable: str = "godot"):
        self.executable = executable

    def available(self) -> bool:
        return shutil.which(self.executable) is not None

    def _run(self, args: list[str], cwd: Path) -> EngineResult:
        proc = subprocess.run(args, cwd=cwd, text=True, capture_output=True, timeout=900)
        return EngineResult(proc.returncode == 0, args, proc.stdout, proc.stderr)

    def run(self, request: ProjectRequest) -> EngineResult:
        return self._run([self.executable, "--path", str(request.project_dir)], request.project_dir)

    def test(self, request: ProjectRequest) -> EngineResult:
        return self._run([self.executable, "--headless", "--path", str(request.project_dir), "--editor", "--quit"], request.project_dir)

    def export(self, request: ProjectRequest, preset: str) -> EngineResult:
        output = request.project_dir / "build" / preset
        output.parent.mkdir(parents=True, exist_ok=True)
        return self._run([self.executable, "--headless", "--path", str(request.project_dir), "--export-release", preset, str(output)], request.project_dir)
