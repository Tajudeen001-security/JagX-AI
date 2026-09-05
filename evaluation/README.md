# JagX Evaluation

JagX capabilities are accepted only when measurable tests pass. Evaluation data must remain isolated from training data.

Evaluation families:
- knowledge and question answering
- reasoning and mathematics
- coding and repository editing
- full-stack and realtime applications
- web animation and graphics
- Godot/Unreal/Unity/custom game generation
- defensive security
- tool use and long-running task recovery
- multimodal tasks as those interfaces are implemented

Every benchmark records task version, model checkpoint, prompt/specification, tool permissions, result, failures and regression status. Use official benchmark methodology when comparing against external models; the local exact-match harness is for reproducible smoke/regression tests, not a claim of frontier performance.
