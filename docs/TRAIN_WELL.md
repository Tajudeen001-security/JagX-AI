# How to train JagX well

Architecture in this repo is ahead of the weights. Intelligence comes from data × tokens × parameters × evaluation.

`jagx train-e2e` is a smoke test (tiny vocab, 2 layers). Do not treat it as training.

## Default serious run (Kaggle T4)

```bash
jagx inspect --config configs/kaggle.json

python scripts/kaggle_train.py \
  --source oasst1 \
  --rows 80000 \
  --steps 3000 \
  --seq-len 512 \
  --batch-size 4 \
  --grad-accum 8 \
  --vocab-size 32000 \
  --hidden-size 512 \
  --layers 8 \
  --heads 8 \
  --context-length 1024 \
  --lr 3e-4 \
  --resume
```

Freeze the tokenizer after the first good 32k vocab. Copy `kaggle_checkpoints/` and `artifacts/kaggle_tokenizer/` off the ephemeral disk every session.

## Scale ladder

1. `configs/kaggle.json` until text is coherent English
2. Mix in `data/seed/gaming_instructions.jsonl` via `scripts/collect_gaming_corpus.py`
3. `scripts/prepare_capability_mixture.py --scale 1.0` for code/math/instruction
4. `configs/kaggle_medium.json` only after val loss is stable
5. `configs/large.json` / `xlarge.json` only on multi-GPU hardware

Token budget: about 20 tokens per parameter (`jagx inspect` prints it).

## After every session

```bash
jagx generate "The agent plans" --checkpoint path/to/step.pt --tokenizer path/to/tok
```

Pass bar: decoded English, no raw `Ġ` pieces, continues the prompt, validation loss trending down.
