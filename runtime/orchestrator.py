from __future__ import annotations

from dataclasses import dataclass, field
from time import monotonic
from typing import Any, Callable, Mapping, Optional
from uuid import uuid4

from .capability_registry import CapabilityRegistry


@dataclass(frozen=True)
class TaskRequest:
    """A normalized request entering the JagX runtime control plane."""

    prompt: str
    capability: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    request_id: str = field(default_factory=lambda: uuid4().hex)


@dataclass(frozen=True)
class TaskResult:
    request_id: str
    capability: str
    status: str
    output: Any = None
    error: Optional[str] = None
    duration_ms: float = 0.0


Handler = Callable[[TaskRequest], Any]


class RuntimeOrchestrator:
    """Route normalized requests to registered native JagX capability handlers.

    The orchestrator deliberately owns routing, limits, receipts, and errors while
    capability implementations remain independent. It has no hosted-model dependency.
    """

    _KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("coding", ("code", "coding", "program", "debug", "repository", "python", "javascript")),
        ("multimodal", ("image", "photo", "audio", "video", "multimodal")),
        ("tools", ("terminal", "command", "tool", "shell")),
        ("memory", ("remember", "memory", "recall")),
        ("games", ("game", "godot", "unity", "unreal")),
        ("finance", ("trade", "trading", "portfolio", "market", "stock")),
        ("security", ("security", "vulnerability", "secure", "threat")),
        ("generation", ("generate", "write", "create", "answer", "explain")),
    )

    def __init__(
        self,
        registry: Optional[CapabilityRegistry] = None,
        *,
        max_prompt_chars: int = 100_000,
        max_retries: int = 1,
    ) -> None:
        if max_prompt_chars <= 0:
            raise ValueError("max_prompt_chars must be positive")
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        self.registry = registry or CapabilityRegistry()
        self.max_prompt_chars = max_prompt_chars
        self.max_retries = max_retries
        self._handlers: dict[str, Handler] = {}

    def register_handler(self, capability: str, handler: Handler) -> None:
        name = capability.strip()
        if not name:
            raise ValueError("capability must not be empty")
        if not callable(handler):
            raise TypeError("handler must be callable")
        self._handlers[name] = handler

    def unregister_handler(self, capability: str) -> None:
        self._handlers.pop(capability, None)

    def classify(self, prompt: str) -> str:
        text = prompt.casefold()
        for capability, keywords in self._KEYWORDS:
            if any(keyword in text for keyword in keywords):
                return capability
        return "generation"

    def dispatch(self, request: TaskRequest) -> TaskResult:
        started = monotonic()
        try:
            if not isinstance(request.prompt, str) or not request.prompt.strip():
                raise ValueError("prompt must be a non-empty string")
            if len(request.prompt) > self.max_prompt_chars:
                raise ValueError("prompt exceeds runtime size limit")

            capability = request.capability.strip() if request.capability else self.classify(request.prompt)
            if not self.registry.available(capability):
                raise RuntimeError(f"capability is unavailable: {capability}")
            handler = self._handlers.get(capability)
            if handler is None:
                raise RuntimeError(f"no handler registered for capability: {capability}")

            last_error: Optional[Exception] = None
            for _ in range(self.max_retries + 1):
                try:
                    output = handler(request)
                    return TaskResult(
                        request_id=request.request_id,
                        capability=capability,
                        status="ok",
                        output=output,
                        duration_ms=(monotonic() - started) * 1000,
                    )
                except Exception as exc:  # capability failures become structured results
                    last_error = exc

            return TaskResult(
                request_id=request.request_id,
                capability=capability,
                status="error",
                error=str(last_error),
                duration_ms=(monotonic() - started) * 1000,
            )
        except Exception as exc:
            capability = request.capability or "unclassified"
            return TaskResult(
                request_id=request.request_id,
                capability=capability,
                status="error",
                error=str(exc),
                duration_ms=(monotonic() - started) * 1000,
            )

    def capabilities(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.registry.list())
