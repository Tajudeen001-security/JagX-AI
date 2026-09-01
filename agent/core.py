from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

from tools.registry import ToolRegistry, ToolResult
from tools.policy import ToolPolicy, require


class Tool(Protocol):
    name: str

    def run(self, arguments: dict) -> dict: ...


@dataclass
class AgentState:
    goal: str
    plan: list[str] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    observations: list[str] = field(default_factory=list)
    step: int = 0
    done: bool = False


class JagXAgent:
    """General-purpose agent with typed tool registry and policy enforcement."""

    def __init__(
        self,
        tools: Optional[list] = None,
        registry: Optional[ToolRegistry] = None,
        policy: Optional[ToolPolicy] = None,
        max_steps: int = 32,
    ):
        self.registry = registry or ToolRegistry()
        self.policy = policy or ToolPolicy()
        self.max_steps = max_steps
        self.audit: list[dict] = []
        if tools:
            for t in tools:
                self.register_legacy(t)

    def register_legacy(self, tool: Tool) -> None:
        """Support older Tool protocol objects."""
        from tools.registry import ToolSpec

        def handler(args: dict) -> ToolResult:
            try:
                data = tool.run(args)
                return ToolResult(ok=True, data=data)
            except Exception as e:
                return ToolResult(ok=False, error=str(e))

        self.registry.register(
            ToolSpec(name=tool.name, description=getattr(tool, "description", tool.name), input_schema={}),
            handler,
        )

    def register(self, name: str, handler, *, description: str = "", permission: str = "filesystem", timeout_s: float = 30.0) -> None:
        from tools.registry import ToolSpec

        def wrapped(args: dict) -> ToolResult:
            try:
                out = handler(args)
                return out if isinstance(out, ToolResult) else ToolResult(ok=True, data=out)
            except Exception as e:
                return ToolResult(ok=False, error=str(e))

        self.registry.register(
            ToolSpec(name=name, description=description or name, input_schema={}, permission=permission, timeout_s=timeout_s),
            wrapped,
        )

    def execute_tool(self, name: str, arguments: dict) -> ToolResult:
        def policy_check(perm: str) -> None:
            require(self.policy, perm)

        result = self.registry.run(name, arguments, policy_check=policy_check)
        self.audit.append({"tool": name, "args_keys": list(arguments.keys()), "ok": result.ok, "error": result.error})
        return result

    def list_tools(self) -> list[str]:
        return [t.name for t in self.registry.list_tools()]
