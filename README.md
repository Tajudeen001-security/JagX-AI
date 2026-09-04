# JagX AI

Independent AI research platform. Core model, tokenizer, training, inference, agent, sandbox, coding loop and evaluation run **without** another AI provider API.

**No demo mode.** Unbound generation returns an error until a real checkpoint is trained and loaded.

## Status

| Area | State |
|------|--------|
| Transformer (RoPE, RMSNorm, SwiGLU, GQA, KV-cache) | Real + tested |
| Tokenizer BPE | Real + tested |
| Training + checkpoint resume | Real + tested |
| Inference from checkpoint | Real (requires your weights) |
| Unified orchestrator | Real + tested |
| Agent TaskDAG | Real + tested |
| Sandbox + coding write→test | Real + tested |
| Memory | Real + tested |
| Local API | Real; generate needs checkpoint |
| Paper trading | Real paper-only |
| Media / multimodal quality | Architecture present; needs trained media weights |

**Not claimed:** frontier chat quality without training.

## Quick start

```bash
pip install -e ".[dev]"
pytest tests/ -q
jagx verify
jagx agent "summarize research notes"
jagx tool echo --args '{"message":"ok"}'
```

Train then generate:

```bash
jagx train --data data.jsonl --tokenizer path/to/tok --config configs/tiny.json --steps 200 --out-dir checkpoints/run1
jagx generate "Hello" --checkpoint checkpoints/run1/... --tokenizer path/to/tok
jagx serve --checkpoint ... --tokenizer ...
```

See `docs/USE_NOW.md`, `docs/FREE_TRAINING.md`, `docs/CAPABILITY_MATRIX.md`.
