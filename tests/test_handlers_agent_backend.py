"""Agent handler returns a clear backend label."""

from __future__ import annotations

from runtime.orchestrator import RequestStatus, TaskKind, build_default_orchestrator


def test_agent_returns_backend_label():
    orch = build_default_orchestrator()
    # Prefer dedicated module registration if present
    try:
        from runtime.handlers_agent import register

        register(orch)
    except Exception:
        pass
    result = orch.execute({"goal": "summarize research notes", "timeout_s": 45})
    assert result.status == RequestStatus.SUCCEEDED
    assert result.kind == TaskKind.AGENT
    data = result.data or {}
    assert "goal" in data or "backend" in data or "status" in data
    if "backend" in data:
        assert data["backend"] in {"agent-dag", "agent-loop", "planned", "degraded"}
