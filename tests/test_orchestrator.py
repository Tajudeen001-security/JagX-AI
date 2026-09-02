"""Unit tests for the unified runtime orchestrator."""

from __future__ import annotations

import time

import pytest

from runtime.orchestrator import (
    Orchestrator,
    RequestStatus,
    StructuredResult,
    TaskKind,
    build_default_orchestrator,
)


def test_classify_basic():
    orch = Orchestrator()
    assert orch.classify({"prompt": "hello"}) == TaskKind.GENERATE
    assert orch.classify({"messages": [{"role": "user", "content": "hi"}]}) == TaskKind.CHAT
    assert orch.classify({"goal": "build a thing"}) == TaskKind.AGENT
    assert orch.classify({"tool": "read_file", "arguments": {}}) == TaskKind.TOOL
    assert orch.classify({"memory_action": "retrieve", "query": "x"}) == TaskKind.MEMORY
    assert orch.classify({"image": b"..."}) == TaskKind.MULTIMODAL
    assert orch.classify({"kind": "code", "instruction": "fix"}) == TaskKind.CODE
    assert orch.classify({"health": True}) == TaskKind.HEALTH


def test_health_handler():
    orch = build_default_orchestrator()
    result = orch.execute({"kind": "health"})
    assert result.status == RequestStatus.SUCCEEDED
    assert result.kind == TaskKind.HEALTH
    assert "uptime_seconds" in result.data
    assert "capabilities" in result.data
    assert result.request_id


def test_generate_unbound():
    orch = build_default_orchestrator()
    result = orch.execute({"prompt": "hello world", "max_tokens": 32})
    assert result.status == RequestStatus.SUCCEEDED
    assert result.kind == TaskKind.GENERATE
    assert result.data["backend"] == "orchestrator-unbound"
    assert result.data["prompt_chars"] == 11
    assert result.latency_ms >= 0


def test_custom_handler_and_request_id():
    orch = Orchestrator()

    def echo(payload, ctx):
        return {"echo": payload.get("text"), "rid": ctx.request_id}

    orch.register_handler(TaskKind.GENERATE, echo)
    rid = "abc123fixed"
    result = orch.execute({"prompt": "x", "text": "hi"}, request_id=rid)
    assert result.request_id == rid
    assert result.status == RequestStatus.SUCCEEDED
    assert result.data["echo"] == "hi"
    assert result.data["rid"] == rid


def test_timeout():
    orch = Orchestrator(default_timeout_s=0.2)

    def slow(payload, ctx):
        time.sleep(1.0)
        return "done"

    orch.register_handler(TaskKind.GENERATE, slow)
    result = orch.execute({"prompt": "slow"}, timeout_s=0.15)
    assert result.status == RequestStatus.TIMED_OUT
    assert result.error_type == "Timeout"


def test_cancel():
    orch = Orchestrator()
    started = {}

    def long_job(payload, ctx):
        started["yes"] = True
        for _ in range(50):
            ctx.check_cancelled()
            time.sleep(0.05)
        return "finished"

    orch.register_handler(TaskKind.AGENT, long_job)

    import threading

    result_box = {}

    def run():
        result_box["r"] = orch.execute({"goal": "long"}, timeout_s=5.0)

    t = threading.Thread(target=run)
    t.start()
    # Wait until active
    for _ in range(40):
        if orch._active:
            break
        time.sleep(0.05)
    assert orch._active
    rid = next(iter(orch._active.keys()))
    assert orch.cancel(rid) is True
    t.join(timeout=3.0)
    r = result_box.get("r")
    assert r is not None
    assert r.status in (RequestStatus.CANCELLED, RequestStatus.FAILED)


def test_memory_add_and_retrieve():
    orch = build_default_orchestrator()
    from memory.store import MemoryStore

    store = MemoryStore()
    add = orch.execute(
        {
            "kind": "memory",
            "memory_action": "add",
            "content": "the capital of France is Paris",
            "durable": False,
            "_store": store,
        }
    )
    assert add.status == RequestStatus.SUCCEEDED
    assert add.data["content"].startswith("the capital")

    hit = orch.execute(
        {
            "kind": "memory",
            "memory_action": "retrieve",
            "query": "capital France",
            "_store": store,
        }
    )
    assert hit.status == RequestStatus.SUCCEEDED
    assert len(hit.data["hits"]) >= 1


def test_limits_rejection():
    from runtime.request_limits import RequestLimits

    orch = Orchestrator(limits=RequestLimits(max_output_tokens=10))
    orch.register_handler(TaskKind.GENERATE, lambda p, c: {"ok": True})
    result = orch.execute({"prompt": "x", "max_tokens": 9999})
    assert result.status == RequestStatus.FAILED
    assert result.error_type == "RequestLimits"


def test_stream_fallback():
    orch = build_default_orchestrator()
    events = list(orch.stream({"prompt": "stream me"}))
    assert events[0]["event"] == "start"
    assert any(e["event"] == "data" for e in events)
    assert events[-1]["event"] == "end"
    assert events[-1]["status"] == RequestStatus.SUCCEEDED.value


def test_structured_result_to_dict():
    r = StructuredResult(
        request_id="r1",
        status=RequestStatus.SUCCEEDED,
        kind=TaskKind.GENERATE,
        data={"text": "hi"},
        latency_ms=1.5,
    )
    d = r.to_dict()
    assert d["request_id"] == "r1"
    assert d["status"] == "succeeded"
    assert d["kind"] == "generate"
    assert d["data"]["text"] == "hi"


def test_no_handler():
    orch = Orchestrator()  # no handlers registered
    result = orch.execute({"prompt": "x"})
    assert result.status == RequestStatus.FAILED
    assert result.error_type == "NoHandler"


def test_audit_events():
    orch = build_default_orchestrator()
    orch.execute({"kind": "health"})
    events = orch.recent_audit(limit=20)
    assert any(e["event"] == "request_received" for e in events)
    assert any(e["event"] == "request_finished" for e in events)
