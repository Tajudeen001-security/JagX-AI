"""API routes wired to the unified orchestrator."""

from __future__ import annotations

from api.server import create_app


def test_execute_route_registered():
    Handler = create_app(use_orchestrator=True)
    assert "/v1/execute" in Handler.post_routes
    assert "/v1/agent" in Handler.post_routes
    assert "/v1/memory" in Handler.post_routes


def test_health_includes_runtime_when_orchestrator_enabled():
    Handler = create_app(use_orchestrator=True)
    payload = Handler.get_routes["/v1/health"]()
    assert payload["status"] == "ok"
    assert "runtime" in payload
    assert "capabilities" in payload["runtime"]


def test_execute_health_kind():
    Handler = create_app(use_orchestrator=True)
    result = Handler.post_routes["/v1/execute"]({"kind": "health"})
    assert result["status"] == "succeeded"
    assert result["kind"] == "health"
    assert "request_id" in result


def test_execute_generate_unbound():
    Handler = create_app(use_orchestrator=True)
    result = Handler.post_routes["/v1/execute"]({"prompt": "hello", "max_tokens": 8})
    assert result["status"] == "succeeded"
    assert result["kind"] == "generate"
    assert result["data"]["backend"] == "orchestrator-unbound"


def test_backward_compat_without_orchestrator():
    Handler = create_app(use_orchestrator=False)
    assert "/v1/execute" not in Handler.post_routes
    assert Handler.get_routes["/health"]()["status"] == "ok"
