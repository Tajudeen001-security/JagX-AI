# JagX AI Capability Matrix

Honest status of each capability. Labels:

- 🟢 Implemented and unit/integration tested
- 🟡 Implemented / experimental / limited (architecture present, quality depends on training or external infra)
- 🔴 Requires unavailable trained models, large compute, or external infrastructure

| Domain | Status | Notes |
|--------|--------|-------|
| Transformer core (RoPE, RMSNorm, SwiGLU, GQA, KV-cache) | 🟢 | Unit tested; no frontier weights |
| Tokenizer (BPE train/load) | 🟢 | Unit tested |
| Local inference + generation controls | 🟢 | Requires checkpoint; unbound API returns 503 |
| Unified runtime orchestrator | 🟢 | Routing, limits, retries, cancel, audit, receipts |
| Agent planner + TaskDAG + DAGExecutor | 🟢 | Heuristic planner + retries + receipts |
| Agent runtime (memory + sandbox tools) | 🟢 | Integrated with orchestrator |
| Coding TaskDAG (inspect → implement → test → repair) | 🟢 | Real sandbox ops when workspace + files provided |
| Tool registry + policy + sandbox | 🟢 | Path boundary, allowlist, timeouts |
| Coding engine (write → test → repair) | 🟢 | Sandboxed; no silent invented patches |
| Memory (short/long, importance, dedupe, session) | 🟢 | Lexical retrieval; embeddings pluggable |
| CLI (`jagx verify|agent|code|train|generate|serve`) | 🟢 | Real commands, no demo mode |
| Local API (`/v1/*`) | 🟢 | generate needs bound weights |
| Multimodal interfaces / encoders / projectors | 🟡 | Architecture + shape tests; no trained multimodal weights |
| Image / audio / video generation backbones | 🟡 | Native pipelines + tests; generation quality not claimed |
| Movie-length orchestration | 🟡 | Scene/shot planning + resume; quality depends on media models |
| Godot / Unity / Unreal adapters | 🟡 | Project scaffolding tested; engines not assumed installed |
| Paper trading + risk controls | 🟢 | Paper-only |
| Defensive security tooling | 🟡 | Prompt-injection scan, sandbox, policy |
| Training pipeline (checkpoint, AMP, resume) | 🟢 | Usable for small runs |
| Evaluation / scorecards / frontier gate | 🟢 | Gates prevent false frontier claims |
| Full-stack web / app generation | 🟡 | Structural generation |
| Frontier-scale intelligence | 🔴 | Not claimed |

Architecture ≠ trained intelligence. No demo facades for generation.
