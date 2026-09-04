from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any] = field(default_factory=dict)
    permission: str = "filesystem"  # maps to ToolPolicy keys
    timeout_s: float = 30.0


@dataclass
class ToolResult:
    ok: bool
    data: Any = None
    error: Optional[str] = None
    audit: dict[str, Any] = field(default_factory=dict)


class ToolRegistry:
    """Typed tool registry with permission and timeout metadata."""

    def __init__(self) -> None:
        self._specs: dict[str, ToolSpec] = {}
        self._handlers: dict[str, Callable[[dict], ToolResult]] = {}

    @classmethod
    def create_default(cls) -> "ToolRegistry":
        reg = cls()
        try:
            from tools.builtin import register_builtin_tools

            register_builtin_tools(reg)
        except Exception:
            pass
        return reg

    def register(self, spec: ToolSpec, handler: Callable[[dict], ToolResult]) -> None:
        self._specs[spec.name] = spec
        self._handlers[spec.name] = handler

    def get(self, name: str) -> ToolSpec:
        if name not in self._specs:
            raise KeyError(f"unknown tool: {name}")
        return self._specs[name]

    def list_tools(self) -> list[ToolSpec]:
        return list(self._specs.values())

    def run(
        self,
        name: str,
        arguments: dict,
        *,
        policy_check: Optional[Callable[[str], None]] = None,
    ) -> ToolResult:
        spec = self.get(name)
        if policy_check is not None:
            policy_check(spec.permission)
        handler = self._handlers[name]
        try:
            result = handler(arguments)
            if not isinstance(result, ToolResult):
                result = ToolResult(ok=True, data=result)
            result.audit = {
                **result.audit,
                "tool": name,
                "permission": spec.permission,
                "timeout_s": spec.timeout_s,
            }
            return result
        except Exception as e:
            return ToolResult(
                ok=False,
                error=str(e),
                audit={"tool": name, "permission": spec.permission},
            )
