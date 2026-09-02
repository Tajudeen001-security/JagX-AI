# JagX AI

Independent AI research platform. Core model, tokenizer, training, inference, agent, sandbox, coding loop and evaluation are designed to run **without** another AI provider API.

## Status (what is implemented and tested)

| Area | State |
|------|--------|
| Transformer (RoPE, RMSNorm, SwiGLU, GQA, KV-cache) | Implemented + unit tested |
| Tokenizer (BPE train/save/load, special tokens) | Implemented + unit tested |
| Training smoke + checkpoint resume | Implemented + unit tested |
| Local inference (checkpoint + tokenizer → generate) | Implemented + smoke tested |
| **Unified runtime orchestrator** (routing, limits, retries, cancel, audit) | Implemented + unit tested |
| Agent + tool registry + policy | Implemented + unit tested |
| **Agent planner + TaskDAG + DAGExecutor** | Implemented + unit tested |
| Sandbox (path boundary, command allowlist, timeout) | Implemented + unit tested |
| Coding engine (write → pytest → repair loop) | Implemented + unit tested |
| Memory (short-term + durable JSONL, importance, session isolation, dedupe) | Implemented + unit tested |
| Local API (`/v1/generate`, `/v1/execute`, `/v1/agent`, `/v1/memory`, health) | Implemented + unit tested |
| Native image/audio/video generator backbones | Implemented + shape/causal-forward tests (not trained models) |
| Godot/Unity/Unreal project adapters | Implemented + project-generation tests (not AAA production) |
| Paper trading + risk controls | Implemented + unit tested; paper-only |
| Movie-length shot planning/resume | Implemented orchestration (generation quality depends on trained media models) |
| Evaluation adapters + frontier claim gate | Implemented + unit tested (no frontier score claim) |

**Not claimed:** trained frontier performance, production-quality image/audio/video generation, or autonomous AAA game production. These require substantial training data, compute, evaluation, and asset/tool integration.

## Quick start

```bash
pip install -e ".[dev]"
pytest tests/ -q
python -c "from evaluation.model_smoke import run_smoke_test; print(run_smoke_test())"
python -c "from runtime.orchestrator import build_default_orchestrator; print(build_default_orchestrator().execute({'kind':'health'}).to_dict())"
```

Configs: `configs/tiny.json`, `configs/small.json`, `configs/medium.json` (architecture only; no trained weights implied).

## Layout

`model/` `tokenizer/` `training/` `inference/` `runtime/` `agent/` `tools/` `coding/` `memory/` `api/` `evaluation/` `multimodal/` `media/` `games/` `capabilities/` `security/` `configs/` `tests/`

See `docs/ARCHITECTURE.md` and `docs/ROADMAP.md`.

External models may be used only as optional research teachers/evaluators. They are never runtime dependencies.

CI keeps linting, tests, and security checks as required gates.
