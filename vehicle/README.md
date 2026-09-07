# JagX Vehicle AI

This package defines a safety-first path for camera-based vehicle intelligence.

## Planned stack
1. Camera/sensor ingestion
2. Perception: vehicles, pedestrians, cyclists, signs, lanes, traffic lights and hazards
3. Localization and mapping
4. Prediction of nearby-agent motion
5. Route and behavior planning
6. Independent safety envelope
7. Vehicle-controller interface
8. Simulation and hardware-in-the-loop evaluation

The current implementation is intentionally **simulation/HIL-oriented**. It does not claim perfect autonomous driving and does not directly control steering, throttle or brakes. A production vehicle system requires redundant sensors, independent safety controllers, validated hardware, extensive closed-course testing, regulatory approval, and a fail-safe path.

Traffic-aware routing, doors, charging, parking, pickup/drop-off and bookings should be implemented as separate host integrations so failure of the language/agent layer cannot directly create an unsafe vehicle action.
