"""Unified AI runtime / control plane for JagX."""

from __future__ import annotations

import time
import traceback
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FuturesTimeout
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterator, Optional

from runtime.capability_registry import Capability, CapabilityRegistry
from runtime.circuit_breaker import CircuitBreaker
from runtime.health import RuntimeHealth
from runtime.quota import TokenBucket
from runtime.request_id import new_request_id
from runtime.request_limits import RequestLimits
from runtime.retry import RetryPolicy


class TaskKind(str, Enum):
    GENERATE = "generate"
    CHAT = "chat"
    AGENT = "agent"
    CODE = "code"
    MEMORY = "memory"
    MULTIMODAL = "multimodal"
    TOOL = "tool"
    HEALTH = "health"
    UNKNOWN = "unknown"


class RequestStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"


@dataclass
class ExecutionContext:
    request_id: str
    session_id: Optional[str] = None
    kind: TaskKind = TaskKind.UNKNOWN
    started_at: float = field(default_factory=time.time)
    deadline: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    cancelled: bool = False

    def remaining_timeout(self) -> Optional[float]:
        if self.deadline is None:
            return None
        return max(0.0, self.deadline - time.time())

    def check_cancelled(self) -> None:
        if self.cancelled:
            raise RuntimeError("request cancelled")


@dataclass
class StructuredResult:
    request_id: str
    status: RequestStatus
    kind: TaskKind
    data: Any = None
    error: Optional[str] = None
    error_type: Optional[str] = None
    latency_ms: float = 0.0
    audit: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "status": self.status.value,
            "kind": self.kind.value,
            "data": self.data,
            "error": self.error,
            "error_type": self.error_type,
            "latency_ms": self.latency_ms,
            "audit": self.audit,
            "metadata": self.metadata,
        }


Handler = Callable[[dict[str, Any], ExecutionContext], Any]
StreamHandler = Callable[[dict[str, Any], ExecutionContext], Iterator[Any]]


class Orchestrator:
    def __init__(self, *, limits=None, retry=None, max_workers: int = 8, default_timeout_s: float = 120.0) -> None:
        self.limits = limits or RequestLimits()
        self.retry = retry or RetryPolicy(attempts=2, base_delay=0.1, max_delay=2.0)
        self.default_timeout_s = default_timeout_s
        self.registry = CapabilityRegistry()
        self.health = RuntimeHealth.start()
        self.breaker = CircuitBreaker(failure_threshold=8, recovery_seconds=20.0)
        self.quota = TokenBucket(capacity=100, refill_per_second=20.0)
        self._handlers: dict[TaskKind, Handler] = {}
        self._stream_handlers: dict[TaskKind, StreamHandler] = {}
        self._active: dict[str, ExecutionContext] = {}
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="jagx-orch")
        self._audit_sink: list[dict[str, Any]] = []
        for name, desc in (
            ("generate", "Local text generation"),
            ("chat", "Chat-style generation"),
            ("agent", "Multi-step agent execution"),
            ("code", "Coding workflow"),
            ("memory", "Memory store / retrieve"),
            ("multimodal", "Image / audio / video paths"),
            ("tool", "Single tool invocation"),
            ("health", "Runtime health snapshot"),
        ):
            self.registry.register(Capability(name=name, description=desc))

    def register_handler(self, kind: TaskKind, handler: Handler) -> None:
        self._handlers[kind] = handler

    def register_stream_handler(self, kind: TaskKind, handler: StreamHandler) -> None:
        self._stream_handlers[kind] = handler

    def classify(self, payload: dict[str, Any]) -> TaskKind:
        explicit = payload.get("kind") or payload.get("task") or payload.get("capability")
        if explicit:
            try:
                return TaskKind(str(explicit).lower())
            except ValueError:
                pass
        if payload.get("health") is True or payload.get("action") == "health":
            return TaskKind.HEALTH
        if "messages" in payload or payload.get("mode") == "chat":
            return TaskKind.CHAT
        if payload.get("goal") or payload.get("agent") is True:
            return TaskKind.AGENT
        if payload.get("code") is not None or payload.get("repo_path") or payload.get("files"):
            return TaskKind.CODE
        if payload.get("memory_action") or (payload.get("query") and payload.get("memory") is True):
            return TaskKind.MEMORY
        if any(k in payload for k in ("image", "audio", "video", "modality", "media")):
            return TaskKind.MULTIMODAL
        if payload.get("tool") or payload.get("tool_name"):
            return TaskKind.TOOL
        if "prompt" in payload or "input" in payload or "text" in payload:
            return TaskKind.GENERATE
        return TaskKind.UNKNOWN

    def _emit(self, event: str, ctx: ExecutionContext, **extra: Any) -> dict[str, Any]:
        record = {"event": event, "request_id": ctx.request_id, "session_id": ctx.session_id, "kind": ctx.kind.value, "ts": time.time(), **extra}
        self._audit_sink.append(record)
        if len(self._audit_sink) > 2000:
            self._audit_sink = self._audit_sink[-1000:]
        return record

    def recent_audit(self, limit: int = 50) -> list[dict[str, Any]]:
        return list(self._audit_sink[-limit:])

    def execute(self, payload: dict[str, Any], *, session_id: Optional[str] = None, timeout_s: Optional[float] = None, request_id: Optional[str] = None) -> StructuredResult:
        if not isinstance(payload, dict):
            raise TypeError("payload must be a dict")
        rid = request_id or new_request_id()
        kind = self.classify(payload)
        timeout = timeout_s if timeout_s is not None else float(payload.get("timeout_s") or self.default_timeout_s)
        ctx = ExecutionContext(request_id=rid, session_id=session_id or payload.get("session_id"), kind=kind, deadline=time.time() + max(0.1, timeout), metadata={"client": payload.get("client")})
        self._active[rid] = ctx
        audit: list[dict[str, Any]] = []
        t0 = time.time()

        def finish(status: RequestStatus, data: Any = None, error: Optional[str] = None, error_type: Optional[str] = None) -> StructuredResult:
            latency = (time.time() - t0) * 1000.0
            ok = status == RequestStatus.SUCCEEDED
            self.health.record(ok)
            (self.breaker.success if ok else self.breaker.failure)()
            self._active.pop(rid, None)
            result = StructuredResult(request_id=rid, status=status, kind=kind, data=data, error=error, error_type=error_type, latency_ms=latency, audit=audit, metadata={"session_id": ctx.session_id})
            audit.append(self._emit("request_finished", ctx, status=status.value, latency_ms=latency, error=error))
            return result

        audit.append(self._emit("request_received", ctx, payload_keys=sorted(payload.keys())))
        if not self.breaker.allow():
            return finish(RequestStatus.FAILED, error="circuit_breaker_open", error_type="CircuitBreaker")
        if not self.quota.consume(1):
            return finish(RequestStatus.FAILED, error="quota_exceeded", error_type="Quota")
        try:
            max_out = int(payload.get("max_tokens") or payload.get("max_new_tokens") or 256)
            approx_in = len(str(payload.get("prompt") or payload.get("input") or payload.get("text") or ""))
            self.limits.validate_text(min(approx_in, self.limits.max_input_tokens), max_out)
        except ValueError as e:
            return finish(RequestStatus.FAILED, error=str(e), error_type="RequestLimits")
        except Exception:
            pass
        handler = self._handlers.get(kind)
        if handler is None:
            if kind == TaskKind.HEALTH:
                return finish(RequestStatus.SUCCEEDED, data=self.health_snapshot())
            return finish(RequestStatus.FAILED, error=f"no handler registered for kind={kind.value}", error_type="NoHandler")

        def operation() -> Any:
            ctx.check_cancelled()
            return handler(payload, ctx)

        try:
            remaining = ctx.remaining_timeout()
            if remaining is not None and remaining <= 0:
                return finish(RequestStatus.TIMED_OUT, error="deadline exceeded before start", error_type="Timeout")
            future: Future = self._executor.submit(lambda: self.retry.run(operation))
            try:
                data = future.result(timeout=remaining)
            except FuturesTimeout:
                ctx.cancelled = True
                future.cancel()
                return finish(RequestStatus.TIMED_OUT, error="execution timed out", error_type="Timeout")
            if ctx.cancelled:
                return finish(RequestStatus.CANCELLED, error="request cancelled", error_type="Cancelled")
            return finish(RequestStatus.SUCCEEDED, data=data)
        except Exception as exc:
            audit.append(self._emit("request_error", ctx, error=str(exc), traceback=traceback.format_exc(limit=6)))
            return finish(RequestStatus.FAILED, error=str(exc), error_type=type(exc).__name__)

    def stream(self, payload: dict[str, Any], *, session_id: Optional[str] = None, timeout_s: Optional[float] = None, request_id: Optional[str] = None) -> Iterator[dict[str, Any]]:
        rid = request_id or new_request_id()
        kind = self.classify(payload)
        timeout = timeout_s if timeout_s is not None else float(payload.get("timeout_s") or self.default_timeout_s)
        ctx = ExecutionContext(request_id=rid, session_id=session_id or payload.get("session_id"), kind=kind, deadline=time.time() + max(0.1, timeout))
        self._active[rid] = ctx
        t0 = time.time()
        yield {"event": "start", "request_id": rid, "kind": kind.value}
        if not self.breaker.allow():
            yield {"event": "error", "request_id": rid, "error": "circuit_breaker_open", "status": RequestStatus.FAILED.value}
            self._active.pop(rid, None)
            return
        handler = self._stream_handlers.get(kind)
        if handler is None:
            result = self.execute(payload, session_id=session_id, timeout_s=timeout, request_id=rid)
            if result.status == RequestStatus.SUCCEEDED:
                yield {"event": "data", "request_id": rid, "data": result.data}
            yield {"event": "end", "request_id": rid, "status": result.status.value, "error": result.error, "latency_ms": result.latency_ms}
            return
        try:
            for chunk in handler(payload, ctx):
                ctx.check_cancelled()
                remaining = ctx.remaining_timeout()
                if remaining is not None and remaining <= 0:
                    yield {"event": "error", "request_id": rid, "error": "timed out", "status": RequestStatus.TIMED_OUT.value}
                    self.health.record(False)
                    self.breaker.failure()
                    self._active.pop(rid, None)
                    return
                yield {"event": "data", "request_id": rid, "data": chunk}
            latency = (time.time() - t0) * 1000.0
            self.health.record(True)
            self.breaker.success()
            yield {"event": "end", "request_id": rid, "status": RequestStatus.SUCCEEDED.value, "latency_ms": latency}
        except Exception as exc:
            latency = (time.time() - t0) * 1000.0
            self.health.record(False)
            self.breaker.failure()
            yield {"event": "error", "request_id": rid, "error": str(exc), "error_type": type(exc).__name__, "status": RequestStatus.FAILED.value, "latency_ms": latency}
        finally:
            self._active.pop(rid, None)

    def cancel(self, request_id: str) -> bool:
        ctx = self._active.get(request_id)
        if ctx is None:
            return False
        ctx.cancelled = True
        self._emit("request_cancelled", ctx)
        return True

    def health_snapshot(self) -> dict[str, Any]:
        snap = self.health.snapshot()
        snap["active_requests"] = len(self._active)
        snap["circuit_open"] = not self.breaker.allow()
        snap["capabilities"] = [{"name": c.name, "enabled": c.enabled, "description": c.description} for c in self.registry.list()]
        return snap

    def shutdown(self, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait)


def build_default_orchestrator() -> Orchestrator:
    orch = Orchestrator()

    def health_handler(payload, ctx):
        return orch.health_snapshot()

    orch.register_handler(TaskKind.HEALTH, health_handler)

    def memory_handler(payload, ctx):
        from memory.store import MemoryStore
        store = payload.get("_store") or MemoryStore()
        action = (payload.get("memory_action") or payload.get("action") or "retrieve").lower()
        if action == "add":
            content = str(payload.get("content") or payload.get("text") or "")
            if not content:
                raise ValueError("memory add requires content")
            rec = store.add(content, kind=str(payload.get("kind") or "episodic"), source=str(payload.get("source") or "orchestrator"), durable=bool(payload.get("durable", False)), metadata={"request_id": ctx.request_id})
            return {"id": rec.id, "content": rec.content}
        if action in ("retrieve", "search", "recall"):
            query = str(payload.get("query") or payload.get("text") or "")
            hits = store.retrieve(query, k=int(payload.get("k") or 5))
            return {"hits": [{"id": h.id, "content": h.content, "kind": h.kind} for h in hits]}
        raise ValueError(f"unsupported memory_action: {action}")

    orch.register_handler(TaskKind.MEMORY, memory_handler)

    def generate_handler(payload, ctx):
        fn = payload.get("_generate_fn")
        if callable(fn):
            return fn(payload)
        prompt = str(payload.get("prompt") or payload.get("input") or payload.get("text") or "")
        return {"object": "jagx.generation", "backend": "orchestrator-unbound", "prompt_chars": len(prompt), "max_tokens": int(payload.get("max_tokens") or payload.get("max_new_tokens") or 64), "text": "", "note": "Bind local generate via _generate_fn", "request_id": ctx.request_id}

    orch.register_handler(TaskKind.GENERATE, generate_handler)
    orch.register_handler(TaskKind.CHAT, generate_handler)

    def tool_handler(payload, ctx):
        from tools.registry import ToolRegistry
        registry = payload.get("_registry") or ToolRegistry()
        name = payload.get("tool") or payload.get("tool_name")
        if not name:
            raise ValueError("tool name required")
        args = payload.get("arguments") or payload.get("args") or {}
        result = registry.run(str(name), args if isinstance(args, dict) else {})
        return {"ok": result.ok, "data": result.data, "error": result.error, "audit": result.audit, "request_id": ctx.request_id}

    orch.register_handler(TaskKind.TOOL, tool_handler)

    def agent_handler(payload, ctx):
        goal = str(payload.get("goal") or payload.get("prompt") or "")
        if not goal:
            raise ValueError("agent requires goal")
        runtime = payload.get("_agent_runtime")
        if runtime is None:
            try:
                from agent.runtime import AgentRuntime
                runtime = AgentRuntime.create(workspace=payload.get("workspace") or payload.get("repo_path"))
            except Exception as exc:
                return {"goal": goal, "status": "degraded", "error": str(exc), "request_id": ctx.request_id}
        if hasattr(runtime, "run_dag"):
            receipt = runtime.run_dag(goal, handlers=payload.get("_dag_handlers"))
            out = receipt.to_dict() if hasattr(receipt, "to_dict") else {"goal": getattr(receipt, "goal", goal), "success": getattr(receipt, "success", False), "duration_s": getattr(receipt, "duration_s", 0.0), "error": getattr(receipt, "error", None), "dag": getattr(receipt, "dag_summary", {})}
            out["request_id"] = ctx.request_id
            out["backend"] = "agent-dag"
            return out
        if hasattr(runtime, "run_goal"):
            result = runtime.run_goal(goal, lambda s=None: [goal], lambda s=None: {"acted": True, "goal": goal}, lambda s=None: True)
            return {"goal": goal, "result": result, "request_id": ctx.request_id, "backend": "agent-loop"}
        return {"goal": goal, "status": "accepted", "request_id": ctx.request_id}

    orch.register_handler(TaskKind.AGENT, agent_handler)

    def code_handler(payload, ctx):
        engine = payload.get("_coding_engine")
        if engine is not None and hasattr(engine, "write_and_test"):
            files = payload.get("files") or {}
            if isinstance(files, dict) and files:
                result = engine.write_and_test(files, test_command=str(payload.get("test_command") or "python3 -m pytest -q"), timeout_s=float(payload.get("timeout_s") or 60.0))
                return {"ok": result.ok, "files_written": result.files_written, "test_output": result.test_output[:4000], "error": result.error, "attempts": result.attempts, "request_id": ctx.request_id, "backend": "coding-engine"}
        instruction = str(payload.get("instruction") or payload.get("prompt") or payload.get("code") or "")
        return {"instruction": instruction[:500], "status": "accepted", "note": "CodingEngine not bound", "request_id": ctx.request_id}

    orch.register_handler(TaskKind.CODE, code_handler)

    def multimodal_handler(payload, ctx):
        modality = payload.get("modality") or ("image" if "image" in payload else "audio" if "audio" in payload else "video" if "video" in payload else "unknown")
        return {"modality": modality, "status": "accepted", "request_id": ctx.request_id}

    orch.register_handler(TaskKind.MULTIMODAL, multimodal_handler)
    return orch
