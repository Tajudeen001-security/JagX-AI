from __future__ import annotations

import shlex

# Commands are deliberately allowlisted. Arbitrary shell execution is never implied by agent access.
DEFAULT_ALLOWED = frozenset(
    {
        "python",
        "python3",
        "pytest",
        "ruff",
        "git",
        "dart",
        "flutter",
        "godot",
        "node",
        "npm",
        "cargo",
        "go",
    }
)


class CommandGuard:
    def __init__(self, allowed=DEFAULT_ALLOWED):
        self.allowed = frozenset(allowed)

    def validate(self, command: str):
        parts = shlex.split(command)
        if not parts:
            raise ValueError("empty command")
        executable = parts[0].rsplit("/", 1)[-1]
        if executable not in self.allowed:
            raise PermissionError(f"command not allowed: {executable}")

        # Block shell composition / redirection regardless of executable.
        if any(x in command for x in ("&&", "||", ";", "|", ">", "<", "`", "$(")):
            raise PermissionError("shell composition/redirection is not allowed")

        # Arbitrary inline code execution is a common agent escape path.
        if executable in {"python", "python3"} and "-c" in parts[1:]:
            raise PermissionError("python -c is not allowed")

        return parts
