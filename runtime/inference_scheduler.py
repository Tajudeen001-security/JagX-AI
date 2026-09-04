from __future__ import annotations

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class InferenceRequest:
    request_id: str
    payload: Any
    submitted_at: float


class InferenceScheduler:
    """Small, dependency-free inference queue for local model serving.

    Requests are bounded by queue size and executed by a worker pool. The
    scheduler intentionally delegates model execution to a caller-provided
    function so it can support CPU, CUDA, quantized, or remote-compatible
    model wrappers without coupling the runtime to one backend.
    """

    def __init__(self, executor_fn: Callable[[Any], Any], *, max_workers: int = 1, max_queue: int = 128):
        if not callable(executor_fn):
            raise TypeError("executor_fn must be callable")
        if max_workers < 1 or max_queue < 1:
            raise ValueError("max_workers and max_queue must be positive")
        self._executor_fn = executor_fn
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="jagx-infer")
        self._slots = threading.BoundedSemaphore(max_queue)
        self._lock = threading.Lock()
        self._active = 0
        self._completed = 0
        self._failed = 0

    def submit(self, request_id: str, payload: Any) -> Future:
        if not request_id:
            raise ValueError("request_id must be non-empty")
        if not self._slots.acquire(blocking=False):
            raise RuntimeError("inference queue is full")
        request = InferenceRequest(request_id, payload, time.time())
        with self._lock:
            self._active += 1

        def run() -> Any:
            try:
                result = self._executor_fn(request.payload)
            except BaseException:
                with self._lock:
                    self._failed += 1
                raise
            else:
                with self._lock:
                    self._completed += 1
                return result
            finally:
                with self._lock:
                    self._active -= 1
                self._slots.release()

        return self._executor.submit(run)

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {"active": self._active, "completed": self._completed, "failed": self._failed}

    def shutdown(self, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait)
