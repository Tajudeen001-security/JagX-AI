from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from tools.sandbox import WorkspaceSandbox


@dataclass
class CodingResult:
    ok: bool
    files_written: list[str] = field(default_factory=list)
    test_output: str = ""
    error: Optional[str] = None
    attempts: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "files_written": list(self.files_written),
            "test_output": self.test_output[:4000],
            "error": self.error,
            "attempts": self.attempts,
        }


class CodingEngine:
    """Bounded code write → test → inspect loop inside a WorkspaceSandbox.

    Does not claim autonomous multi-file repo repair yet. Provides the verified
    execution substrate so an agent can iterate safely.
    """

    def __init__(self, sandbox: WorkspaceSandbox, max_repair_attempts: int = 3):
        self.sandbox = sandbox
        self.max_repair_attempts = max_repair_attempts

    def write_file(self, relative: str, content: str) -> Path:
        return self.sandbox.write(relative, content)

    def read_file(self, relative: str) -> str:
        return self.sandbox.read(relative)

    def run_tests(self, command: str = "python3 -m pytest -q", timeout_s: float = 60.0) -> dict:
        return self.sandbox.run_command(command, timeout_s=timeout_s)

    def write_and_test(
        self,
        files: dict[str, str],
        test_command: str = "python3 -m pytest -q",
        timeout_s: float = 60.0,
    ) -> CodingResult:
        written = []
        for rel, content in files.items():
            self.write_file(rel, content)
            written.append(rel)
        result = self.run_tests(test_command, timeout_s=timeout_s)
        return CodingResult(
            ok=bool(result.get("ok")),
            files_written=written,
            test_output=(result.get("stdout") or "") + (result.get("stderr") or ""),
            error=None if result.get("ok") else result.get("stderr") or "tests failed",
            attempts=1,
        )

    def repair_loop(
        self,
        files: dict[str, str],
        repair_fn,
        test_command: str = "python3 -m pytest -q",
        timeout_s: float = 60.0,
    ) -> CodingResult:
        """Apply repair_fn(files, test_output) -> new_files up to max_repair_attempts."""
        current = dict(files)
        last = CodingResult(ok=False, error="no attempts")
        for attempt in range(1, self.max_repair_attempts + 1):
            last = self.write_and_test(current, test_command=test_command, timeout_s=timeout_s)
            last.attempts = attempt
            if last.ok:
                return last
            current = repair_fn(current, last.test_output)
            if not isinstance(current, dict):
                last.error = "repair_fn must return dict[str,str]"
                return last
        return last

    def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Orchestrator-friendly entrypoint.

        Expected payload keys:
        - files: dict[str, str] relative path -> content
        - test_command: optional
        - timeout_s: optional
        - repair: optional bool; if true and repair fails, return last result
        """
        files = payload.get("files") or {}
        if not isinstance(files, dict) or not files:
            instruction = str(payload.get("instruction") or payload.get("prompt") or payload.get("code") or "")
            return {
                "instruction": instruction[:500],
                "status": "accepted",
                "note": "CodingEngine.run requires files dict",
            }
        test_command = str(payload.get("test_command") or "python3 -m pytest -q")
        timeout_s = float(payload.get("timeout_s") or 60.0)
        result = self.write_and_test(files, test_command=test_command, timeout_s=timeout_s)
        out = result.to_dict()
        out["backend"] = "coding-engine"
        out["status"] = "ok" if result.ok else "failed"
        return out
