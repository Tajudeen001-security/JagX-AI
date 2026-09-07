"""Safety-first autonomy contracts.

This layer is intentionally simulator/controller agnostic. It can consume
perception from JagX vision models and emit a constrained driving intent for a
verified vehicle controller. It is not a road-ready autonomous driving stack.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol


class Maneuver(str, Enum):
    STOP = "stop"
    FOLLOW = "follow"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    LANE_KEEP = "lane_keep"
    PULL_OVER = "pull_over"


@dataclass(frozen=True)
class VehicleState:
    speed_mps: float
    lane_offset_m: float
    route_remaining_m: float
    hazards: tuple[str, ...] = ()


@dataclass(frozen=True)
class DrivingIntent:
    maneuver: Maneuver
    target_speed_mps: float
    reason: str


class Controller(Protocol):
    def apply(self, intent: DrivingIntent) -> None: ...


class SafetyEnvelope:
    """Independent last-line constraints before a controller receives intent."""

    def __init__(self, max_speed_mps: float = 30.0):
        if max_speed_mps <= 0:
            raise ValueError("max_speed_mps must be positive")
        self.max_speed_mps = max_speed_mps

    def constrain(self, state: VehicleState, intent: DrivingIntent) -> DrivingIntent:
        if state.hazards:
            return DrivingIntent(Maneuver.STOP, 0.0, "hazard detected by perception")
        speed = max(0.0, min(intent.target_speed_mps, self.max_speed_mps))
        return DrivingIntent(intent.maneuver, speed, intent.reason)


@dataclass
class AutonomyPlanner:
    safety: SafetyEnvelope = field(default_factory=SafetyEnvelope)

    def plan(self, state: VehicleState) -> DrivingIntent:
        if state.hazards:
            raw = DrivingIntent(Maneuver.STOP, 0.0, "hazard")
        elif state.route_remaining_m <= 0:
            raw = DrivingIntent(Maneuver.PULL_OVER, 0.0, "route complete")
        elif abs(state.lane_offset_m) > 1.5:
            raw = DrivingIntent(Maneuver.LANE_KEEP, min(state.speed_mps, 8.0), "large lane offset")
        else:
            raw = DrivingIntent(Maneuver.FOLLOW, min(max(state.speed_mps, 2.0), 20.0), "normal route following")
        return self.safety.constrain(state, raw)
