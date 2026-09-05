# JagX-AI Kaggle GPU Training

## One-time Kaggle setup

1. Create/open a Kaggle Notebook.
2. Turn **Internet** on.
3. Set **Accelerator = GPU** (T4/P100 or another available CUDA GPU).
4. Upload/import `notebooks/jagx_kaggle_train.ipynb` from this repository.
5. Run the cells from top to bottom.

The notebook clones the current `main` branch, installs JagX plus the Hugging Face dataset client, downloads 50,000 OpenAssistant/oasst1 records in streaming mode, prepares a deterministic 95/5 train/validation split, removes exact duplicates, trains a BPE tokenizer, verifies that at least one fixed-size batch can be produced, and only then starts native CUDA training.

## Default training run

- Corpus: 80,000 OASST1 records
- Validation: 5%
- Sequence length: 512
- Batch size: 4
- Gradient accumulation: 8
- Model: 8 layers / 512 hidden / 8 heads / 4 KV heads (`configs/kaggle.json`)
- Optimizer: AdamW
- Learning rate: 3e-4
- Steps: 3,000
- Attention: PyTorch SDPA when available

These defaults are a bounded research run, not a frontier-model training run. Do not use `jagx train-e2e` as the training path. Increase corpus size, model size, and training steps only after the end-to-end pipeline completes successfully. Mix `data/seed/gaming_instructions.jsonl` when you want the gaming specialist track.

## Changing the run

Set environment variables before the training cell, for example:

```python
import os
os.environ['JAGX_ROWS'] = '80000'
os.environ['JAGX_STEPS'] = '2000'
```

The launcher fails early if CUDA is unavailable or if the prepared corpus produces no training batch, preventing the previous `training batches are empty` failure from reaching the trainer.

## Outputs

Kaggle working storage receives:

- `data/raw/oasst1-kaggle-*.jsonl`
- `data/prepared/train.jsonl`
- `data/prepared/validation.jsonl`
- `data/prepared/tokenizer_corpus.txt`
- `artifacts/kaggle_tokenizer/`
- `kaggle_checkpoints/`

Do not commit these generated datasets or checkpoints to GitHub. The repository keeps source manifests and reproducible download/preparation code instead.
