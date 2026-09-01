from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

from safety.prompt_injection import scan_prompt
from tools.registry import ToolRegistry, ToolResult
from tools.policy import ToolPolicy, PolicyError, require


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
    """General-purpose agent with typed tool registry, policy, and injection gating."""

    def __init__(
        self,
        tools: Optional[list] = None,
        registry: Optional[ToolRegistry] = None,
        policy: Optional[ToolPolicy] = None,
        max_steps: int = 32,
        gate_injections: bool = True,
    ):
        self.registry = registry or ToolRegistry()
        self.policy = policy or ToolPolicy()
        self.max_steps = max_steps
        self.gate_injections = gate_injections
        self.audit: list[dict] = []
        if tools:
            for t in tools:
                self.register_legacy(t)

    def register_legacy(self, tool: Tool) -> None:
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

    def register(
        self,
        name: str,
        handler,
        *,
        description: str = "",
        permission: str = "filesystem",
        timeout_s: float = 30.0,
    ) -> None:
        from tools.registry import ToolSpec

        def wrapped(args: dict) -> ToolResult:
            try:
                out = handler(args)
                return out if isinstance(out, ToolResult) else ToolResult(ok=True, data=out)
            except Exception as e:
                return ToolResult(ok=False, error=str(e))

        self.registry.register(
            ToolSpec(
                name=name,
                description=description or name,
                input_schema={},
                permission=permission,
                timeout_s=timeout_s,
            ),
            wrapped,
        )

    def _check_injection(self, arguments: dict) -> None:
        if not self.gate_injections:
            return
        # Scan string argument values for injection patterns before elevated tools.
        for key, value in arguments.items():
            if isinstance(value, str):
                scan = scan_prompt(value)
                if scan.flagged:
                    raise PolicyError(
                        f"prompt-injection patterns in tool args[{key}]: {','.join(scan.patterns)}"
                    )

    def execute_tool(self, name: str, arguments: dict) -> ToolResult:
        def policy_check(perm: str) -> None:
            require(self.policy, perm)

        try:
            # Always policy-check; injection gate for any non-read permission path
            spec = self.registry.get(name)
            if spec.permission in {"shell", "network", "deployment", "security_testing"}:
                self._check_injection(arguments)
            elif self.gate_injections:
                # Also scan filesystem writes that embed user text
                self._check_injection(arguments)

            result = self.registry.run(name, arguments, policy_check=policy_check)
        except PolicyError as e:
            result = ToolResult(ok=False, error=str(e), audit={"tool": name, "blocked": True})
        except KeyError as e:
            result = ToolResult(ok=False, error=str(e))

        self.audit.append(
            {"tool": name, "args_keys": list(arguments.keys()), "ok": result.ok, "error": result.error}
        )
        return result

    def list_tools(self) -> list[str]:
        return [t.name for t in self.registry.list_tools()]
