from dataclasses import dataclass, field
from typing import Any, Callable

@dataclass
class RuntimeState:
    task_id: str
    goal: str
    step: int = 0
    artifacts: list[str] = field(default_factory=list)
    observations: list[str] = field(default_factory=list)
    checkpoints: list[dict[str, Any]] = field(default_factory=list)

class AgentRuntime:
    """Provider-independent runtime for long-running engineering tasks."""
    def __init__(self, infer: Callable[[str], str], tools=None):
        self.infer=infer
        self.tools={t.name:t for t in (tools or [])}
    def register(self,tool): self.tools[tool.name]=tool
    def checkpoint(self,state:RuntimeState):
        item={"step":state.step,"artifacts":list(state.artifacts),"observations":list(state.observations)}
        state.checkpoints.append(item); return item
