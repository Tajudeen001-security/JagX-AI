from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ExecutionResult:
    task_id: str
    status: str
    result: Any = None
    error: str | None = None
    attempts: int = 0
    elapsed_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class ExecutionEngine:
    """Small, deterministic task executor for orchestrator/agent plans."""

    def execute(
        self,
        task_id: str,
        fn: Callable[[], Any],
        *,
        retries: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> ExecutionResult:
        if not task_id.strip():
            raise ValueError("task_id must be non-empty")
        if retries < 0:
            raise ValueError("retries must be non-negative")
        started = time.perf_counter()
        attempts = 0
        last_error: Exception | None = None
        for _ in range(retries + 1):
            attempts += 1
            try:
                value = fn()
                return ExecutionResult(task_id, "succeeded", value, attempts=attempts,
                                       elapsed_ms=(time.perf_counter() - started) * 1000,
                                       metadata=dict(metadata or {}))
            except Exception as exc:  # task boundary: preserve failure for the caller
                last_error = exc
        return ExecutionResult(task_id, "failed", attempts=attempts,
                               error=str(last_error),
                               elapsed_ms=(time.perf_counter() - started) * 1000,
                               metadata=dict(metadata or {}))
