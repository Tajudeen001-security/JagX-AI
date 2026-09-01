# JagX AI

Independent AI research platform. Core model, tokenizer, training, inference, agent, sandbox, coding loop and evaluation are designed to run **without** another AI provider API.

## Status (what is implemented and tested)

| Area | State |
|------|--------|
| Transformer (RoPE, RMSNorm, SwiGLU, GQA, KV-cache) | Implemented + unit tested |
| Tokenizer (BPE train/save/load, special tokens) | Implemented + unit tested |
| Training smoke + checkpoint resume | Implemented + unit tested |
| Local inference (checkpoint + tokenizer → generate) | Implemented + smoke tested |
| Agent + tool registry + policy | Implemented + unit tested |
| Sandbox (path boundary, command allowlist, timeout) | Implemented + unit tested |
| Coding engine (write → pytest → repair loop) | Implemented + unit tested |
| Memory (short-term + durable JSONL retrieve) | Implemented + unit tested |
| Multimodal contracts + encoder interfaces | Interfaces + shape tests (not trained models) |
| Evaluation adapters | Real model execution smoke (not frontier scores) |

**Not claimed:** trained frontier performance, image/video generation quality, or autonomous AAA game production.

## Quick start

```bash
pip install -e ".[dev]"
pytest tests/ -q
python -c "from evaluation.model_smoke import run_smoke_test; print(run_smoke_test())"
```

Configs: `configs/tiny.json`, `configs/small.json`, `configs/medium.json` (architecture only; no trained weights implied).

## Layout

`model/` `tokenizer/` `training/` `inference/` `agent/` `tools/` `coding/` `memory/` `evaluation/` `multimodal/` `media/` `security/` `configs/` `tests/`

See `docs/ARCHITECTURE.md` and `docs/ROADMAP.md`.

External models may be used only as optional research teachers/evaluators. They are never runtime dependencies.
