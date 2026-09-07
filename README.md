# JagX AI

Independent AI research platform. Core model, tokenizer, training, inference, agent, sandbox, coding loop and evaluation run **without** another AI provider API.

**No demo mode.** Unbound generation returns an error until a real checkpoint is trained and loaded.

Scale configs now go from a 10M smoke model to ~3B (`configs/kaggle.json` through `configs/xlarge.json`). Bigger JSON is not a trained brain — run `jagx inspect` then train.

## Status

| Area | State |
|------|-------|
| Transformer (RoPE, RMSNorm, SwiGLU, GQA, KV-cache, SDPA) | Real + tested |
| Tokenizer BPE | Real + tested |
| Training + checkpoint resume | Real + tested |
| Scale ladder (tiny → xlarge) + `jagx inspect` | Real + tested |
| Inference from checkpoint | Real (requires your weights) |
| Unified orchestrator | Real + tested |
| Agent TaskDAG | Real + tested |
| Sandbox + coding write→test | Real + tested |
| Memory | Real + tested |
| Task experience learning store | Real + tested |
| Bounded computer-control contract | Real + tested; requires local host adapter |
| Vision | Trainable image encoder + perception contracts |
| Multilingual routing | Real routing contract; language quality depends on trained data/models |
| Vehicle autonomy | Safety-first simulation/planning contracts; **not road-ready** |
| Gaming instruction seed corpus | Real MIT seed + collector |
| Local API | Real; generate needs checkpoint |
| Paper trading | Real paper-only |
| Media / multimodal quality | Architecture present; needs trained media weights |

**Not claimed:** frontier chat quality, universal African-language fluency, or production autonomous driving without the required trained models, evaluation and safety validation.

## New agent architecture

JagX now separates the model from the capabilities around it:

- `agent/computer_control.py` — explicit user-granted screen, cursor, keyboard, terminal and filesystem control contracts.
- `memory/learning.py` — persistent task experience that can be retrieved without retraining model weights after every task.
- `multimodal/perception.py` — camera/screen/video perception adapter contracts.
- `multilingual/router.py` — automatic response-language routing with English fallback.
- `autonomy/vehicle.py` — perception-to-planning interfaces and an independent safety envelope for future simulation/robotics work.

Computer access is deny-by-default at the integration boundary. `FULL_CONTROL` is available only when the host application explicitly grants it, and destructive operations still request confirmation by default.

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

Inspect a size and train on Kaggle (not the e2e toy):

```bash
jagx inspect --config configs/kaggle.json
python scripts/collect_gaming_corpus.py
python scripts/kaggle_train.py --resume
```

See `docs/TRAIN_WELL.md`, `docs/KAGGLE_TRAINING.md`, `docs/USE_NOW.md`, `docs/FREE_TRAINING.md`, `docs/CAPABILITY_MATRIX.md`.
