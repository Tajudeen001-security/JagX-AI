# JagX AI Capability Roadmap v2

This is the implementation direction for the next JagX generation.

## 1. Multilingual African intelligence
- Automatic text language detection.
- Automatic speech-language detection.
- Text ↔ speech continuity.
- Extensible African-language registry targeting 2,000+ languages/dialects.
- Language-specific evaluation and low-resource data pipelines.
- Generate websites, code, games, documents and media in the user's selected language.

## 2. Vision and camera
- Image understanding now has a camera-stream abstraction.
- Add mobile/web camera adapters.
- Add video-frame sampling and object detection.
- Add OCR, scene understanding and temporal reasoning.

## 3. Learning
- Collect approved corrections and task outcomes.
- Build versioned datasets from feedback.
- Train/evaluate new checkpoints separately.
- Never silently rewrite deployed weights.

## 4. Real-world tools
- Search and research.
- Traffic-aware route research.
- Booking discovery.
- Reservation only after explicit confirmation.
- Company/contact messaging only after explicit confirmation.
- Computer control remains deny-by-default and permission-gated.

## 5. Games, apps and media
- Godot, Unreal and Unity project agents.
- Web/Android/iOS/desktop project generation.
- Image, audio, animation and video workflows.
- Documents and export pipelines.

## 6. Vehicle AI research
JagX will have a dedicated vehicle stack for simulation and hardware-in-the-loop work:

`camera/sensors → perception → localization → prediction → route planning → behavior planning → independent safety envelope → controller interface`

It can reason about stopping, following, turning, lane changes, traffic signals, hazards, parking, pickup/drop-off and traffic-aware routing. The current agent must not directly actuate a real vehicle. Real deployment requires independent safety controls, redundancy, simulation, closed-course validation and regulatory compliance.

## 7. Scaling
The architecture remains compatible with the current millions-of-parameters experiments and later larger checkpoints. Model size alone is not the objective: data quality, reasoning, tool use, multilingual coverage, evaluation and safety determine capability.
