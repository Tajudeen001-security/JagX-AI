"""Production-oriented agent planner with task DAG and dependency resolution.

Complex goals are decomposed into executable subtasks with explicit
dependencies, retry budgets, state tracking and execution receipts.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


class TaskState(str, Enum):
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


@dataclass
class TaskNode:
    """Single node in a task DAG."""

    id: str
    name: str
    description: str = ""
    depends_on: list[str] = field(default_factory=list)
    state: TaskState = TaskState.PENDING
    retry_budget: int = 2
    attempts: int = 0
    result: Any = None
    error: Optional[str] = None
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def receipt(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "state": self.state.value,
            "attempts": self.attempts,
            "error": self.error,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "depends_on": list(self.depends_on),
            "metadata": dict(self.metadata),
        }


@dataclass
class TaskDAG:
    """Directed acyclic graph of agent tasks with dependency resolution."""

    nodes: dict[str, TaskNode] = field(default_factory=dict)
    goal: str = ""
    created_at: float = field(default_factory=time.time)

    def add(
        self,
        name: str,
        *,
        description: str = "",
        depends_on: Optional[list[str]] = None,
        retry_budget: int = 2,
        node_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        validate_deps: bool = True,
    ) -> TaskNode:
        tid = node_id or uuid.uuid4().hex[:12]
        if tid in self.nodes:
            raise ValueError(f"duplicate task id: {tid}")
        deps = list(depends_on or [])
        if validate_deps:
            for dep in deps:
                if dep not in self.nodes:
                    raise ValueError(f"task depends on missing task {dep}")
        node = TaskNode(
            id=tid,
            name=name,
            description=description,
            depends_on=deps,
            retry_budget=retry_budget,
            metadata=metadata or {},
        )
        self.nodes[tid] = node
        return node

    def validate(self) -> None:
        """Ensure all dependency edges exist and the graph is acyclic."""
        for node in self.nodes.values():
            for dep in node.depends_on:
                if dep not in self.nodes:
                    raise ValueError(f"task {node.id} depends on missing task {dep}")
        # Cycle detection via DFS
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {tid: WHITE for tid in self.nodes}

        def dfs(tid: str) -> None:
            color[tid] = GRAY
            for dep in self.nodes[tid].depends_on:
                if color[dep] == GRAY:
                    raise ValueError(f"cycle detected involving {tid} -> {dep}")
                if color[dep] == WHITE:
                    dfs(dep)
            color[tid] = BLACK

        for tid in self.nodes:
            if color[tid] == WHITE:
                dfs(tid)

    def ready_tasks(self) -> list[TaskNode]:
        ready = []
        for node in self.nodes.values():
            if node.state not in (TaskState.PENDING, TaskState.READY):
                continue
            deps_ok = all(
                self.nodes[d].state == TaskState.SUCCEEDED for d in node.depends_on
            )
            if deps_ok:
                node.state = TaskState.READY
                ready.append(node)
            elif any(self.nodes[d].state in (TaskState.FAILED, TaskState.CANCELLED) for d in node.depends_on):
                node.state = TaskState.SKIPPED
                node.error = "dependency failed or cancelled"
        return ready

    def all_terminal(self) -> bool:
        terminal = {
            TaskState.SUCCEEDED,
            TaskState.FAILED,
            TaskState.SKIPPED,
            TaskState.CANCELLED,
        }
        return all(n.state in terminal for n in self.nodes.values())

    def summary(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for n in self.nodes.values():
            counts[n.state.value] = counts.get(n.state.value, 0) + 1
        return {
            "goal": self.goal,
            "total": len(self.nodes),
            "counts": counts,
            "receipts": [n.receipt() for n in self.nodes.values()],
        }


class Planner:
    """Decompose a goal into a TaskDAG.

    Default strategy is heuristic keyword-based decomposition suitable for
    coding / research / multi-step workflows. A model-backed planner can be
    injected via plan_fn without changing the execution substrate.
    """

    def __init__(self, plan_fn: Optional[Callable[[str], TaskDAG]] = None) -> None:
        self.plan_fn = plan_fn

    def plan(self, goal: str) -> TaskDAG:
        if self.plan_fn is not None:
            dag = self.plan_fn(goal)
            dag.goal = goal
            dag.validate()
            return dag
        return self._default_plan(goal)

    def _default_plan(self, goal: str) -> TaskDAG:
        g = goal.lower()
        dag = TaskDAG(goal=goal)

        # Always start with understand + plan
        understand = dag.add("understand", description=f"Parse and clarify goal: {goal[:200]}")
        plan_node = dag.add("plan", description="Produce ordered subtasks", depends_on=[understand.id])

        if any(k in g for k in ("code", "implement", "fix", "refactor", "test", "bug")):
            inspect = dag.add("inspect_repo", description="Inspect workspace / repository", depends_on=[plan_node.id])
            implement = dag.add("implement", description="Write or modify code", depends_on=[inspect.id], retry_budget=3)
            test = dag.add("run_tests", description="Execute tests in sandbox", depends_on=[implement.id], retry_budget=2)
            repair = dag.add("repair", description="Repair failures from tests", depends_on=[test.id], retry_budget=3)
            dag.add("summarize", description="Summarize coding outcome", depends_on=[repair.id])
        elif any(k in g for k in ("image", "audio", "video", "generate media", "multimodal")):
            prep = dag.add("prepare_media", description="Validate media request", depends_on=[plan_node.id])
            gen = dag.add("generate_media", description="Run media generation pipeline", depends_on=[prep.id], retry_budget=2)
            dag.add("validate_output", description="Validate generated media", depends_on=[gen.id])
        elif any(k in g for k in ("trade", "portfolio", "backtest", "finance")):
            data = dag.add("load_market_data", description="Load paper market data", depends_on=[plan_node.id])
            risk = dag.add("check_risk", description="Apply risk limits", depends_on=[data.id])
            exec_n = dag.add("paper_execute", description="Paper-trade execution", depends_on=[risk.id])
            dag.add("report", description="Portfolio / PnL report", depends_on=[exec_n.id])
        elif any(k in g for k in ("security", "vulnerability", "audit", "threat")):
            scope = dag.add("scope_security", description="Define defensive security scope", depends_on=[plan_node.id])
            analyze = dag.add("static_analysis", description="Run static / dependency analysis", depends_on=[scope.id])
            dag.add("security_report", description="Produce remediation suggestions", depends_on=[analyze.id])
        else:
            # Generic research / multi-step
            gather = dag.add("gather", description="Gather relevant context / memory", depends_on=[plan_node.id])
            act = dag.add("act", description="Execute primary action", depends_on=[gather.id], retry_budget=2)
            verify = dag.add("verify", description="Verify outcome", depends_on=[act.id])
            dag.add("summarize", description="Summarize result", depends_on=[verify.id])

        dag.validate()
        return dag


@dataclass
class ExecutionReceipt:
    goal: str
    dag_summary: dict[str, Any]
    success: bool
    duration_s: float
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "success": self.success,
            "duration_s": self.duration_s,
            "error": self.error,
            "dag": self.dag_summary,
        }


class DAGExecutor:
    """Execute a TaskDAG with retry budgets, cancellation and receipts."""

    def __init__(
        self,
        handlers: Optional[dict[str, Callable[[TaskNode, dict[str, Any]], Any]]] = None,
        *,
        max_parallel: int = 1,
        on_event: Optional[Callable[[str, TaskNode], None]] = None,
    ) -> None:
        self.handlers = handlers or {}
        self.max_parallel = max_parallel
        self.on_event = on_event
        self._cancelled = False
        self.context: dict[str, Any] = {}

    def cancel(self) -> None:
        self._cancelled = True

    def register(self, name: str, handler: Callable[[TaskNode, dict[str, Any]], Any]) -> None:
        self.handlers[name] = handler

    def _emit(self, event: str, node: TaskNode) -> None:
        if self.on_event:
            self.on_event(event, node)

    def run(self, dag: TaskDAG, *, default_handler: Optional[Callable[[TaskNode, dict[str, Any]], Any]] = None) -> ExecutionReceipt:
        t0 = time.time()
        self._cancelled = False
        error: Optional[str] = None

        while not dag.all_terminal():
            if self._cancelled:
                for n in dag.nodes.values():
                    if n.state in (TaskState.PENDING, TaskState.READY, TaskState.RUNNING):
                        n.state = TaskState.CANCELLED
                        n.error = "cancelled"
                error = "execution cancelled"
                break

            ready = dag.ready_tasks()
            if not ready:
                # Deadlock or all remaining blocked
                for n in dag.nodes.values():
                    if n.state in (TaskState.PENDING, TaskState.READY):
                        n.state = TaskState.FAILED
                        n.error = n.error or "no runnable path"
                break

            # Sequential by default (max_parallel=1); safe for sandbox side effects
            for node in ready[: self.max_parallel]:
                node.state = TaskState.RUNNING
                node.started_at = time.time()
                node.attempts += 1
                self._emit("task_start", node)
                handler = self.handlers.get(node.name) or default_handler
                try:
                    if handler is None:
                        # No-op success for unspecified steps (planner scaffold)
                        result = {"status": "noop", "task": node.name}
                    else:
                        result = handler(node, self.context)
                    node.result = result
                    node.state = TaskState.SUCCEEDED
                    node.finished_at = time.time()
                    # Allow handlers to stash shared context
                    if isinstance(result, dict) and "context_update" in result:
                        self.context.update(result["context_update"])
                    self._emit("task_success", node)
                except Exception as exc:
                    node.error = str(exc)
                    if node.attempts <= node.retry_budget:
                        node.state = TaskState.PENDING  # retry later
                        self._emit("task_retry", node)
                    else:
                        node.state = TaskState.FAILED
                        node.finished_at = time.time()
                        self._emit("task_failed", node)

        success = all(
            n.state in (TaskState.SUCCEEDED, TaskState.SKIPPED)
            for n in dag.nodes.values()
        ) and not any(n.state == TaskState.FAILED for n in dag.nodes.values())
        # More precise: succeed if no FAILED and at least one SUCCEEDED
        failed = any(n.state == TaskState.FAILED for n in dag.nodes.values())
        succeeded = any(n.state == TaskState.SUCCEEDED for n in dag.nodes.values())
        success = succeeded and not failed and not self._cancelled

        return ExecutionReceipt(
            goal=dag.goal,
            dag_summary=dag.summary(),
            success=success,
            duration_s=time.time() - t0,
            error=error,
        )
