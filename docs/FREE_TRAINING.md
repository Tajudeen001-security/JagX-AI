# Free training for years (without the stack breaking)

Honest guide. **Free GPU time exists. Unlimited free frontier training does not.**

## What "lifetime free" can actually mean

| Goal | Realistic on free tiers? |
|------|---------------------------|
| Train **tiny/small** JagX configs for years | Yes, if you checkpoint and resume |
| Improve a **small** model weekly | Yes (Kaggle ~30 GPU-h/week, Colab variable) |
| Match GPT-class frontier models for $0 forever | **No** |
| Keep the **code + checkpoints** working for years | Yes (this is what we design for) |

Free platforms change quotas. Design for **resume**, not one long run.

## Free GPU sources (2026-style)

Rotate so one platform dying does not stop you:

1. **Kaggle** — often ~30 GPU hours/week after phone verify; sessions ~9–12h. Best stable free quota.
2. **Google Colab free** — T4 when available; ~12h sessions; quota not guaranteed.
3. **Lightning AI free credits** — limited monthly credits; good for uninterrupted short runs.
4. **Your own CPU** — slow but unlimited for `configs/tiny.json` experiments.

Chain them: train on Kaggle → save checkpoint to Drive/HF/Git LFS → resume on Colab next day.

## Rules so training does not "break" over years

1. **Always checkpoint** (model + optimizer + step + RNG + config).
2. **Resume from checkpoint** — never rely on a single 12h session finishing everything.
3. **Version data** — JSONL with `source`, `license`, `split`, `quality` fields (JagX data contract).
4. **Pin software** — record `torch` / `tokenizers` versions in your run notes.
5. **Start small** — `configs/tiny.json` → `small.json` only when loss is stable.
6. **Do not claim frontier** until evaluation gates say so (repo already has a frontier claim gate).

## Minimal free workflow

```bash
# 1) Install (Colab/Kaggle/local)
pip install -e ".[dev]"
# install torch for the platform GPU/CPU as needed

# 2) Train tokenizer on your open data (once)
# python -m tokenizer.train_tokenizer ...

# 3) Train with resume-friendly entrypoint
python -m training.entrypoint \
  --data your_train.jsonl \
  --tokenizer path/to/tokenizer \
  --config configs/tiny.json \
  --steps 500 \
  --out-dir checkpoints/run1 \
  --device cuda   # or cpu

# 4) Next free session — resume
python -m training.entrypoint \
  --data your_train.jsonl \
  --tokenizer path/to/tokenizer \
  --config configs/tiny.json \
  --resume checkpoints/run1/... \
  --steps 500 \
  --out-dir checkpoints/run1
```

Store checkpoints outside the notebook ephemeral disk (Google Drive, Hugging Face Hub, GitHub Releases for small files).

## Data (legal, free, long-term)

Use **open-licensed** text/code only (e.g. public domain, permissive OSS with attribution).  
Do not scrape copyrighted books/sites blindly. Provenance is required in JagX training docs.

## What free training will produce

- A **working** small JagX model that runs fully offline.
- Improving quality only as **data + hours** accumulate.
- **Not** automatic ChatGPT-level intelligence from free tiers alone.

## If platforms cut free GPU

Your repo still works: CPU training, smaller batches, community shared checkpoints (when you publish open weights under a clear license).
