from __future__ import annotations

from .planner import VehiclePlanner


def run_smoke_scenarios() -> list[dict[str, object]]:
    planner = VehiclePlanner()
    scenarios = [
        {"name": "clear_road", "distance": None, "closing": False, "speed": 10.0, "route": "proceed"},
        {"name": "near_obstacle", "distance": 4.0, "closing": False, "speed": 10.0, "route": "proceed"},
        {"name": "closing_obstacle", "distance": 15.0, "closing": True, "speed": 10.0, "route": "proceed"},
        {"name": "turn", "distance": None, "closing": False, "speed": 6.0, "route": "turn_left"},
    ]
    results = []
    for s in scenarios:
        command = planner.decide(
            obstacle_distance_m=s["distance"],
            obstacle_closing=s["closing"],
            requested_speed_mps=s["speed"],
            route_action=s["route"],
        )
        results.append({"name": s["name"], "command": command.__dict__})
    return results


if __name__ == "__main__":
    for result in run_smoke_scenarios():
        print(result)
