"""Agent capability handler for the unified orchestrator."""

from __future__ import annotations

from typing import Any

from runtime.orchestrator import ExecutionContext


def agent_handler(payload: dict[str, Any], ctx: ExecutionContext) -> dict[str, Any]:
    """Execute multi-step agent goals via AgentRuntime.run_dag when available."""
    goal = str(payload.get("goal") or payload.get("prompt") or "")
    if not goal:
        raise ValueError("agent requires goal")

    runtime = payload.get("_agent_runtime")
    if runtime is None:
        try:
            from agent.runtime import AgentRuntime

            workspace = payload.get("workspace") or payload.get("repo_path")
            runtime = AgentRuntime.create(workspace=workspace)
        except Exception:
            runtime = None

    files = payload.get("files") if isinstance(payload.get("files"), dict) else None
    test_command = str(payload.get("test_command") or "python3 -m pytest -q")
    timeout_s = float(payload.get("timeout_s") or 60.0)
    repair_fn = payload.get("_repair_fn")

    if runtime is not None and hasattr(runtime, "run_dag"):
        handlers = payload.get("_dag_handlers")
        receipt = runtime.run_dag(
            goal,
            handlers=handlers,
            files=files,
            test_command=test_command,
            timeout_s=timeout_s,
            repair_fn=repair_fn if callable(repair_fn) else None,
        )
        out = {
            "goal": goal,
            "success": receipt.success,
            "duration_s": receipt.duration_s,
            "dag": receipt.dag_summary,
            "error": receipt.error,
            "request_id": ctx.request_id,
            "backend": "agent-dag",
        }
        return out

    if runtime is not None and hasattr(runtime, "run_goal"):

        def plan_fn(*_a, **_k):
            return [goal]

        def act_fn(*_a, **_k):
            return {"acted": True, "goal": goal}

        def verify_fn(*_a, **_k) -> bool:
            return True

        out = runtime.run_goal(goal, plan_fn, act_fn, verify_fn)
        return {
            "goal": goal,
            "result": out,
            "request_id": ctx.request_id,
            "backend": "agent-loop",
        }

    try:
        from agent.planner import Planner

        dag = Planner().plan(goal)
        return {
            "goal": goal,
            "status": "planned",
            "dag": dag.summary(),
            "note": "AgentRuntime unavailable; returned plan only",
            "request_id": ctx.request_id,
            "backend": "planned",
        }
    except Exception as exc:
        return {
            "goal": goal,
            "status": "accepted",
            "note": f"Agent path limited: {exc}",
            "request_id": ctx.request_id,
            "backend": "degraded",
        }


def register(orch) -> None:
    from runtime.orchestrator import TaskKind

    orch.register_handler(TaskKind.AGENT, agent_handler)
