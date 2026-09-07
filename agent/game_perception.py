"""Game-screen perception primitives for the JagX game agent.

The module is model/provider agnostic: a host can supply screenshots or frames,
then connect an actual vision model later. It deliberately does not implement
anti-cheat bypasses or hidden input injection.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class ScreenFrame:
    image: Any
    timestamp: float
    width: int
    height: int


@dataclass
class GameState:
    game_id: str | None = None
    scene: str | None = None
    player: dict[str, Any] = field(default_factory=dict)
    objectives: list[str] = field(default_factory=list)
    ui: list[dict[str, Any]] = field(default_factory=list)
    hazards: list[dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)


class GameVisionAdapter(Protocol):
    def perceive(self, frame: ScreenFrame) -> GameState: ...


class GamePerception:
    """Normalizes observations before the planner sees them."""

    def __init__(self, adapter: GameVisionAdapter):
        self.adapter = adapter
        self.history: list[GameState] = []

    def observe(self, frame: ScreenFrame) -> GameState:
        state = self.adapter.perceive(frame)
        self.history.append(state)
        if len(self.history) > 32:
            self.history.pop(0)
        return state
