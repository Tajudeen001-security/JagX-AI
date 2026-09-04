from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from agent.core import AgentState, JagXAgent
from agent.loop import AgentLoop
from agent.planner import DAGExecutor, ExecutionReceipt, Planner, TaskDAG, TaskNode
from memory import MemoryStore
from tools.policy import ToolPolicy
from tools.sandbox import WorkspaceSandbox


@dataclass
class AgentRuntime:
    """Combines agent, memory, sandbox, loop and DAG planner for task execution."""

    agent: JagXAgent
    memory: MemoryStore
    sandbox: Optional[WorkspaceSandbox] = None
    loop: AgentLoop = field(default_factory=lambda: AgentLoop(max_steps=16, max_retries=1))
    planner: Planner = field(default_factory=Planner)

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
        """Legacy plan→act→verify loop (backward compatible)."""
        state = AgentState(goal=goal)
        self.remember(f"goal: {goal}")
        result = self.loop.run(plan_fn, act_fn, verify_fn)
        self.remember(f"result: {str(result)[:500]}")
        state.done = True
        return result

    def run_dag(
        self,
        goal: str,
        *,
        handlers: Optional[dict[str, Callable[[TaskNode, dict[str, Any]], Any]]] = None,
        default_handler: Optional[Callable[[TaskNode, dict[str, Any]], Any]] = None,
        dag: Optional[TaskDAG] = None,
        files: Optional[dict[str, str]] = None,
        test_command: str = "python3 -m pytest -q",
        timeout_s: float = 60.0,
        repair_fn: Optional[Callable[[dict[str, str], str], dict[str, str]]] = None,
    ) -> ExecutionReceipt:
        """Plan (or accept) a TaskDAG and execute with receipts, retries and memory."""
        self.remember(f"goal: {goal}")
        if dag is None:
            dag = self.planner.plan(goal)
        else:
            dag.goal = goal
            dag.validate()

        resolved: dict[str, Callable[[TaskNode, dict[str, Any]], Any]] = dict(handlers or {})

        # Real coding handlers when a workspace sandbox is available
        if self.sandbox is not None:
            from agent.coding_handlers import build_coding_handlers

            coding = build_coding_handlers(
                self.sandbox,
                files=files,
                test_command=test_command,
                timeout_s=timeout_s,
                repair_fn=repair_fn,
            )
            for name, fn in coding.items():
                resolved.setdefault(name, fn)

        if "gather" not in resolved:

            def gather_handler(node: TaskNode, ctx: dict[str, Any]) -> dict[str, Any]:
                hits = self.recall(goal, k=5)
                ctx["memory_hits"] = hits
                return {"hits": hits, "context_update": {"memory_hits": hits}}

            resolved["gather"] = gather_handler

        if "summarize" not in resolved:

            def summarize_handler(node: TaskNode, ctx: dict[str, Any]) -> dict[str, Any]:
                summary = {
                    "goal": goal,
                    "memory_hits": ctx.get("memory_hits", []),
                    "notes": f"Completed DAG for: {goal[:200]}",
                }
                self.remember(f"summary: {summary['notes']}", durable=False)
                return summary

            resolved["summarize"] = summarize_handler

        # Seed context for coding files if provided
        executor = DAGExecutor(handlers=resolved)
        if files:
            executor.context["files"] = dict(files)
        executor.context["goal"] = goal

        receipt = executor.run(dag, default_handler=default_handler)
        self.remember(
            f"dag_result: success={receipt.success} duration_s={receipt.duration_s:.3f}",
            durable=False,
        )
        return receipt
