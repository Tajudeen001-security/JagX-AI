from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentStep:
    action: str
    tool: str | None = None
    arguments: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""


@dataclass
class AgentPlan:
    goal: str
    steps: list[AgentStep]

    def validate(self) -> "AgentPlan":
        if not self.goal.strip():
            raise ValueError("agent goal cannot be empty")
        if not self.steps:
            raise ValueError("agent plan must contain at least one step")
        for step in self.steps:
            if step.action not in {"reason", "tool", "verify", "respond"}:
                raise ValueError(f"unsupported agent action: {step.action}")
            if step.action == "tool" and not step.tool:
                raise ValueError("tool step requires a tool name")
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "steps": [
                {"action": s.action, "tool": s.tool, "arguments": s.arguments, "rationale": s.rationale}
                for s in self.steps
            ],
        }
