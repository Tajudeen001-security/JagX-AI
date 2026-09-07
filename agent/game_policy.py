"""Rule-aware game action policy contracts.

The policy produces ordinary game actions and leaves execution to the
user-authorized computer/controller adapter. It must not bypass anti-cheat,
DRM, authentication, or other platform protections.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from .game_perception import GameState


@dataclass(frozen=True)
class GameAction:
    kind: str
    value: Any = None
    duration_ms: int = 0
    reason: str = ""


@dataclass
class GamePlan:
    actions: list[GameAction] = field(default_factory=list)
    objective: str = ""
    confidence: float = 0.0


class GamePolicy(Protocol):
    def plan(self, state: GameState, objective: str) -> GamePlan: ...


class RuleAwarePolicy:
    """Conservative fallback policy used until a trained game policy exists."""

    def plan(self, state: GameState, objective: str) -> GamePlan:
        if state.confidence < 0.5:
            return GamePlan(objective=objective, confidence=state.confidence)
        return GamePlan(objective=objective, confidence=state.confidence)
