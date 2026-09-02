from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Capability:
    name: str
    enabled: bool = True
    description: str = ""
    dependencies: tuple[str, ...] = field(default_factory=tuple)


class CapabilityRegistry:
    """Central runtime registry for native JagX capabilities."""

    def __init__(self) -> None:
        self._items: dict[str, Capability] = {}

    def register(self, capability: Capability) -> None:
        if not capability.name.strip():
            raise ValueError("capability name must not be empty")
        self._items[capability.name] = capability

    def enable(self, name: str) -> None:
        self._items[name].enabled = True

    def disable(self, name: str) -> None:
        self._items[name].enabled = False

    def available(self, name: str) -> bool:
        item = self._items.get(name)
        return bool(item and item.enabled)

    def list(self) -> tuple[Capability, ...]:
        return tuple(self._items.values())
