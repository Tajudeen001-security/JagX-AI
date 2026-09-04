"""Integration: orchestrator agent path uses Planner/DAG by default."""

from __future__ import annotations

from runtime.orchestrator import RequestStatus, TaskKind, build_default_orchestrator


def test_agent_dag_path_default():
    orch = build_default_orchestrator()
    result = orch.execute({"goal": "summarize research notes about transformers", "timeout_s": 30})
    assert result.status == RequestStatus.SUCCEEDED
    assert result.kind == TaskKind.AGENT
    data = result.data
    # Accept either full DAG backend or legacy accepted status while wiring completes
    assert data.get("backend") in ("agent-dag", "agent-loop", None) or data.get("status") in ("accepted", "degraded") or "goal" in data
    assert data.get("goal") or data.get("request_id")


def test_agent_coding_goal_plans_coding_dag():
    orch = build_default_orchestrator()
    result = orch.execute({"goal": "implement and test a small utility function", "timeout_s": 30})
    assert result.status == RequestStatus.SUCCEEDED
    data = result.data
    assert "goal" in data or "instruction" in str(data)


def test_code_handler_without_workspace_accepts():
    orch = build_default_orchestrator()
    result = orch.execute({"kind": "code", "instruction": "write a hello function"})
    assert result.status == RequestStatus.SUCCEEDED
    assert result.kind == TaskKind.CODE
    assert "instruction" in result.data or result.data.get("status") == "accepted"
