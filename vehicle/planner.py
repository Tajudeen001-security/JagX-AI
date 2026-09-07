from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SafetyEnvelope:
    max_speed_mps: float = 13.9
    minimum_stop_distance_m: float = 8.0
    emergency_stop: bool = True


@dataclass(frozen=True)
class DriveCommand:
    action: str  # stop, slow, follow, turn_left, turn_right, proceed
    target_speed_mps: float
    reason: str


class VehiclePlanner:
    """Conservative decision layer intended for simulation/HIL testing.

    It does not directly actuate a vehicle. A validated vehicle controller must
    consume commands and enforce independent braking, steering, redundancy,
    and watchdog limits.
    """

    def __init__(self, envelope: SafetyEnvelope | None = None):
        self.envelope = envelope or SafetyEnvelope()

    def decide(self, *, obstacle_distance_m: float | None,
               obstacle_closing: bool = False,
               requested_speed_mps: float = 0.0,
               route_action: str = "proceed") -> DriveCommand:
        if self.envelope.emergency_stop and obstacle_distance_m is not None:
            if obstacle_distance_m <= self.envelope.minimum_stop_distance_m:
                return DriveCommand("stop", 0.0, "safety envelope: obstacle too close")
            if obstacle_closing:
                return DriveCommand("slow", min(requested_speed_mps, 5.0), "closing obstacle")

        speed = max(0.0, min(requested_speed_mps, self.envelope.max_speed_mps))
        if route_action not in {"stop", "slow", "follow", "turn_left", "turn_right", "proceed"}:
            return DriveCommand("stop", 0.0, "unknown route action")
        if route_action == "stop":
            speed = 0.0
        return DriveCommand(route_action, speed, "route planner")
