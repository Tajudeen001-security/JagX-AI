from runtime.capability_registry import Capability
from runtime.orchestrator import RuntimeOrchestrator, TaskRequest


def test_orchestrator_routes_registered_capability():
    runtime = RuntimeOrchestrator()
    runtime.registry.register(Capability("coding"))
    runtime.register_handler("coding", lambda request: f"handled: {request.prompt}")

    result = runtime.dispatch(TaskRequest("debug this code"))

    assert result.status == "ok"
    assert result.capability == "coding"
    assert result.output == "handled: debug this code"
    assert result.request_id
    assert result.duration_ms >= 0


def test_orchestrator_classifies_common_requests():
    runtime = RuntimeOrchestrator()
    assert runtime.classify("remember my family recipe") == "memory"
    assert runtime.classify("inspect this vulnerability") == "security"
    assert runtime.classify("make a Godot game") == "games"
    assert runtime.classify("generate an image") == "multimodal"


def test_orchestrator_returns_structured_handler_errors_and_retries():
    runtime = RuntimeOrchestrator(max_retries=2)
    runtime.registry.register(Capability("generation"))
    attempts = {"count": 0}

    def fail(_request):
        attempts["count"] += 1
        raise RuntimeError("backend unavailable")

    runtime.register_handler("generation", fail)
    result = runtime.dispatch(TaskRequest("answer this"))

    assert result.status == "error"
    assert result.error == "backend unavailable"
    assert attempts["count"] == 3


def test_orchestrator_rejects_empty_prompt():
    runtime = RuntimeOrchestrator()
    result = runtime.dispatch(TaskRequest("   "))
    assert result.status == "error"
    assert "non-empty" in result.error
