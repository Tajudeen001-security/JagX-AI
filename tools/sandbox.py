from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Optional

from security.command_guard import CommandGuard


class WorkspaceSandbox:
    """Filesystem + bounded command execution sandbox.

    - Paths cannot escape the root workspace.
    - Commands must pass CommandGuard allowlist.
    - Hard timeout and output size limits.
    - No network by default (caller must not grant it).
    """

    def __init__(
        self,
        root: str | Path,
        *,
        command_guard: Optional[CommandGuard] = None,
        default_timeout_s: float = 30.0,
        max_output_bytes: int = 256_000,
    ):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.guard = command_guard or CommandGuard()
        self.default_timeout_s = default_timeout_s
        self.max_output_bytes = max_output_bytes
        self.audit_log: list[dict] = []

    def path(self, relative: str | Path) -> Path:
        target = (self.root / relative).resolve()
        if target != self.root and self.root not in target.parents:
            raise ValueError("Path escapes sandbox")
        return target

    def write(self, relative: str | Path, content: str) -> Path:
        target = self.path(relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        # Python can reuse a stale .pyc when rapid repair iterations land in
        # the same filesystem timestamp tick. Invalidate the sibling cache so
        # an agent always tests the source it just wrote.
        if target.suffix == ".py":
            cache = target.parent / "__pycache__"
            stem = target.stem
            if cache.is_dir():
                for compiled in cache.glob(f"{stem}.*.pyc"):
                    try:
                        compiled.unlink()
                    except OSError:
                        pass
        self.audit_log.append({"op": "write", "path": str(relative), "bytes": len(content.encode("utf-8"))})
        return target

    def read(self, relative: str | Path) -> str:
        target = self.path(relative)
        data = target.read_text(encoding="utf-8")
        self.audit_log.append({"op": "read", "path": str(relative), "bytes": len(data.encode("utf-8"))})
        return data

    def list_dir(self, relative: str | Path = ".") -> list[str]:
        target = self.path(relative)
        if not target.is_dir():
            raise NotADirectoryError(str(relative))
        return sorted(p.name for p in target.iterdir())

    def run_command(
        self,
        command: str,
        *,
        timeout_s: Optional[float] = None,
        cwd: Optional[str] = None,
        env: Optional[dict] = None,
    ) -> dict:
        parts = self.guard.validate(command)
        timeout = timeout_s if timeout_s is not None else self.default_timeout_s
        workdir = self.path(cwd) if cwd else self.root

        clean_env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": str(self.root),
            "LANG": "C.UTF-8",
        }
        if env:
            clean_env.update(env)

        started = time.time()
        try:
            proc = subprocess.run(
                parts,
                cwd=str(workdir),
                env=clean_env,
                capture_output=True,
                timeout=timeout,
                check=False,
            )
            stdout = proc.stdout[: self.max_output_bytes].decode("utf-8", errors="replace")
            stderr = proc.stderr[: self.max_output_bytes].decode("utf-8", errors="replace")
            record = {
                "op": "run_command",
                "command": parts,
                "returncode": proc.returncode,
                "duration_s": time.time() - started,
                "stdout_len": len(stdout),
                "stderr_len": len(stderr),
            }
            self.audit_log.append(record)
            return {
                "ok": proc.returncode == 0,
                "returncode": proc.returncode,
                "stdout": stdout,
                "stderr": stderr,
                "duration_s": record["duration_s"],
            }
        except subprocess.TimeoutExpired:
            self.audit_log.append({"op": "run_command", "command": parts, "error": "timeout", "timeout_s": timeout})
            return {
                "ok": False,
                "returncode": -1,
                "stdout": "",
                "stderr": f"timeout after {timeout}s",
                "duration_s": timeout,
            }
