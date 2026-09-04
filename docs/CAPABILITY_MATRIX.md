# JagX AI Capability Matrix

Honest status of each capability. Labels:

- 🟢 Implemented and unit/integration tested
- 🟡 Implemented / experimental / limited (architecture present, quality depends on training or external infra)
- 🔴 Requires unavailable trained models, large compute, or external infrastructure

| Domain | Status | Notes |
|--------|--------|-------|
| Transformer core (RoPE, RMSNorm, SwiGLU, GQA, KV-cache) | 🟢 | Unit tested; no frontier weights |
| Tokenizer (BPE train/load) | 🟢 | Unit tested |
| Local inference + generation controls | 🟢 | Smoke tested; quality = untrained / tiny configs |
| Unified runtime orchestrator | 🟢 | Routing, limits, retries, cancel, audit, receipts |
| Agent planner + TaskDAG + DAGExecutor | 🟢 | Heuristic planner + retries + receipts |
| Agent runtime (memory + sandbox tools) | 🟢 | Integrated with orchestrator |
| Tool registry + policy + sandbox | 🟢 | Path boundary, allowlist, timeouts |
| Coding engine (write → test → repair) | 🟢 | Sandboxed; not autonomous multi-file SOTA |
| Memory (short/long, importance, dedupe, session) | 🟢 | Lexical retrieval; embeddings pluggable |
| Local API (`/v1/*`) | 🟢 | Health, generate, execute, agent, memory |
| Multimodal interfaces / encoders / projectors | 🟡 | Architecture + shape tests; no trained multimodal weights |
| Image / audio / video generation backbones | 🟡 | Native pipelines + tests; generation quality not claimed |
| Movie-length orchestration | 🟡 | Scene/shot planning + resume; quality depends on media models |
| Godot / Unity / Unreal adapters | 🟡 | Project scaffolding tested; engines not assumed installed |
| Paper trading + risk controls | 🟢 | Paper-only; real money behind explicit auth (not enabled) |
| Defensive security tooling | 🟡 | Prompt-injection scan, sandbox, policy; not full SAST suite |
| Training pipeline (checkpoint, AMP, resume) | 🟢 | Usable for small runs; no fabricated pretrained weights |
| Evaluation / scorecards / frontier gate | 🟢 | Gates prevent false frontier claims |
| Full-stack web / app generation | 🟡 | Structural generation; quality not production-guaranteed |
| Frontier-scale intelligence | 🔴 | Not claimed; requires data + compute + evaluation |

Every capability must retain tests. Architecture ≠ trained intelligence.
