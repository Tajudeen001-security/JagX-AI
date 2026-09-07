"""User-authorized computer-control primitives for JagX.

A host application supplies a local adapter for screen/cursor/keyboard/
terminal access. JagX cannot use a hidden remote-control channel.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Protocol


class AccessLevel(str, Enum):
    READ_ONLY = "read_only"
    APP_ONLY = "app_only"
    WORKSPACE = "workspace"
    FULL_CONTROL = "full_control"


class ComputerAction(str, Enum):
    SCREENSHOT = "screenshot"
    MOVE_CURSOR = "move_cursor"
    CLICK = "click"
    TYPE = "type"
    SCROLL = "scroll"
    KEY = "key"
    OPEN_APP = "open_app"
    READ_TERMINAL = "read_terminal"
    RUN_COMMAND = "run_command"
    READ_FILE = "read_file"
    WRITE_FILE = "write_file"
    DELETE_FILE = "delete_file"


@dataclass(frozen=True)
class ComputerRequest:
    action: ComputerAction
    target: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    reason: str = ""


@dataclass
class AccessGrant:
    level: AccessLevel
    workspace: Path | None = None
    allowed_apps: set[str] = field(default_factory=set)
    expires_at: float | None = None
    require_confirmation_for_destructive: bool = True

    def allows(self, request: ComputerRequest) -> bool:
        if self.level == AccessLevel.READ_ONLY:
            return request.action in {ComputerAction.SCREENSHOT, ComputerAction.READ_TERMINAL, ComputerAction.READ_FILE}
        if self.level == AccessLevel.APP_ONLY:
            if request.action in {ComputerAction.SCREENSHOT, ComputerAction.READ_TERMINAL}:
                return True
            return request.target in self.allowed_apps
        if self.level == AccessLevel.WORKSPACE:
            if request.action in {ComputerAction.SCREENSHOT, ComputerAction.READ_TERMINAL, ComputerAction.RUN_COMMAND}:
                return True
            if request.action in {ComputerAction.READ_FILE, ComputerAction.WRITE_FILE, ComputerAction.DELETE_FILE}:
                if self.workspace is None:
                    return False
                try:
                    Path(request.target).resolve().relative_to(self.workspace.resolve())
                    return True
                except ValueError:
                    return False
            return False
        return True

    def needs_confirmation(self, request: ComputerRequest) -> bool:
        return self.require_confirmation_for_destructive and request.action in {
            ComputerAction.DELETE_FILE,
            ComputerAction.RUN_COMMAND,
            ComputerAction.OPEN_APP,
        }


class ComputerAdapter(Protocol):
    def execute(self, request: ComputerRequest) -> dict[str, Any]: ...


class ComputerController:
    def __init__(self, adapter: ComputerAdapter, grant: AccessGrant):
        self.adapter = adapter
        self.grant = grant
        self.audit: list[dict[str, Any]] = []

    def execute(self, request: ComputerRequest, *, confirmed: bool = False) -> dict[str, Any]:
        if not self.grant.allows(request):
            raise PermissionError(f"computer action outside granted boundary: {request.action.value}")
        if self.grant.needs_confirmation(request) and not confirmed:
            return {"ok": False, "needs_confirmation": True, "action": request.action.value, "target": request.target}
        result = self.adapter.execute(request)
        self.audit.append({"action": request.action.value, "target": request.target, "ok": bool(result.get("ok", True))})
        return result
