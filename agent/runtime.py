from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from agent.core import AgentState, JagXAgent
from agent.loop import AgentLoop
from memory import MemoryStore
from tools.policy import ToolPolicy
from tools.sandbox import WorkspaceSandbox


@dataclass
class AgentRuntime:
    """Combines agent, memory, sandbox and loop for task execution."""

    agent: JagXAgent
    memory: MemoryStore
    sandbox: Optional[WorkspaceSandbox] = None
    loop: AgentLoop = field(default_factory=lambda: AgentLoop(max_steps=16, max_retries=1))

    @classmethod
    def create(
        cls,
        workspace: Optional[str] = None,
        policy: Optional[ToolPolicy] = None,
        memory_path: Optional[str] = None,
    ) -> "AgentRuntime":
        policy = policy or ToolPolicy(filesystem=True, shell=False)
        agent = JagXAgent(policy=policy)
        memory = MemoryStore(memory_path)
        sandbox = WorkspaceSandbox(workspace) if workspace else None

        if sandbox is not None:
            def read_file(args: dict) -> Any:
                return sandbox.read(args["path"])

            def write_file(args: dict) -> Any:
                sandbox.write(args["path"], args["content"])
                return {"written": args["path"]}

            agent.register("read_file", read_file, description="Read a file in workspace", permission="filesystem")
            agent.register("write_file", write_file, description="Write a file in workspace", permission="filesystem")

        return cls(agent=agent, memory=memory, sandbox=sandbox)

    def remember(self, content: str, *, durable: bool = False, kind: str = "episodic") -> None:
        self.memory.add(content, durable=durable, kind=kind, source="agent")

    def recall(self, query: str, k: int = 5) -> list[str]:
        return [r.content for r in self.memory.retrieve(query, k=k)]

    def run_goal(self, goal: str, plan_fn, act_fn, verify_fn) -> Any:
        state = AgentState(goal=goal)
        self.remember(f"goal: {goal}")
        result = self.loop.run(plan_fn, act_fn, verify_fn)
        self.remember(f"result: {str(result)[:500]}")
        state.done = True
        return result
