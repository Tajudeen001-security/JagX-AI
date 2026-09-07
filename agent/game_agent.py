"""Game-playing orchestration for user-authorized automation.

This module plans and executes gameplay through the existing computer-control
boundary. It is intended for games that permit automation, offline games,
local simulations, and the user's own games. It does not implement anti-cheat
bypass, credential theft, or unauthorized online-game automation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

from .computer_control import ComputerAction, ComputerController, ComputerRequest


class GameObjective(str, Enum):
    PLAY = "play"
    FINISH_LEVEL = "finish_level"
    HIGH_SCORE = "high_score"
    PRACTICE = "practice"
    PLAYTEST = "playtest"


@dataclass
class GameState:
    frame: Any = None
    objective: GameObjective = GameObjective.PLAY
    score: float | None = None
    progress: float | None = None
    terminal: bool = False
    success: bool = False
    hazards: list[str] = field(default_factory=list)
    available_actions: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GameAction:
    action: ComputerAction
    target: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""


class GamePerception(Protocol):
    def observe(self) -> GameState: ...


class GamePlanner(Protocol):
    def plan(self, state: GameState) -> list[GameAction]: ...


class GameAgent:
    """Observe -> plan -> act -> observe gameplay loop."""

    def __init__(self, computer: ComputerController, perception: GamePerception, planner: GamePlanner):
        self.computer = computer
        self.perception = perception
        self.planner = planner
        self.history: list[dict[str, Any]] = []

    def step(self) -> GameState:
        state = self.perception.observe()
        if state.terminal:
            return state

        actions = self.planner.plan(state)
        for planned in actions:
            request = ComputerRequest(
                action=planned.action,
                target=planned.target,
                payload=planned.payload,
                reason=planned.rationale or "gameplay objective",
            )
            result = self.computer.execute(request)
            self.history.append({
                "action": planned.action.value,
                "target": planned.target,
                "result": result,
            })
            if result.get("needs_confirmation"):
                break
        return self.perception.observe()

    def run(self, max_steps: int = 1000) -> GameState:
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        state = self.perception.observe()
        for _ in range(max_steps):
            if state.terminal or state.success:
                break
            state = self.step()
        return state
