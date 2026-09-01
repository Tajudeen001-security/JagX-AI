from __future__ import annotations
from dataclasses import dataclass
import time


@dataclass
class RuntimeHealth:
    started_at: float
    requests: int = 0
    failures: int = 0

    @classmethod
    def start(cls):
        return cls(time.time())

    def record(self, ok: bool):
        self.requests += 1
        if not ok:
            self.failures += 1

    def snapshot(self):
        uptime = max(0.0, time.time() - self.started_at)
        return {
            "uptime_seconds": uptime,
            "requests": self.requests,
            "failures": self.failures,
            "failure_rate": self.failures / self.requests if self.requests else 0.0,
        }
