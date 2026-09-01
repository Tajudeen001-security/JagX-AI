from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass(frozen=True)
class AuditEvent:
    action: str
    allowed: bool
    reason: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class AuditLog:
    def __init__(self):
        self.events: list[AuditEvent] = []

    def record(self, action: str, allowed: bool, reason: str = "") -> AuditEvent:
        event = AuditEvent(action, allowed, reason)
        self.events.append(event)
        return event

    def denied(self) -> list[AuditEvent]:
        return [event for event in self.events if not event.allowed]
