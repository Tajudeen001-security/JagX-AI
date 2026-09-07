"""Permission boundary for JagX computer and project agents."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Risk(str, Enum):
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    EXTERNAL = "external"
    DESTRUCTIVE = "destructive"


@dataclass(frozen=True)
class ActionRequest:
    tool: str
    target: str
    risk: Risk
    description: str


class PermissionPolicy:
    """Deny-by-default policy for host control.

    Integrators should persist approvals per session and surface the exact
    target/action to the user before granting write, execute, external, or
    destructive operations.
    """

    APPROVAL_REQUIRED = {
        Risk.WRITE,
        Risk.EXECUTE,
        Risk.EXTERNAL,
        Risk.DESTRUCTIVE,
    }

    def requires_approval(self, request: ActionRequest) -> bool:
        return request.risk in self.APPROVAL_REQUIRED

    def describe(self, request: ActionRequest) -> str:
        return f"{request.tool}: {request.description} -> {request.target}"
