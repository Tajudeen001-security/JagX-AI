from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any


class WorkspaceTool:
    """Restricted local workspace operations for development agents.

    The caller supplies an explicit workspace root. Paths are resolved inside
    that root; shell=True and network access are intentionally not exposed.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _safe(self, path: str) -> Path:
        candidate = (self.root / path).resolve()
        if candidate != self.root and self.root not in candidate.parents:
            raise PermissionError("path escapes agent workspace")
        return candidate

    def read_file(self, path: str, max_bytes: int = 1_000_000) -> str:
        target = self._safe(path)
        data = target.read_bytes()
        return data[:max_bytes].decode("utf-8", errors="replace")

    def write_file(self, path: str, content: str) -> str:
        target = self._safe(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return str(target.relative_to(self.root))

    def run_test_command(self, argv: list[str], timeout: int = 60) -> dict[str, Any]:
        if not argv or any("\x00" in x for x in argv):
            raise ValueError("invalid command")
        proc = subprocess.run(
            argv,
            cwd=self.root,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
            shell=False,
        )
        return {"returncode": proc.returncode, "output": proc.stdout[-100_000:]}
