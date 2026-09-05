"""Paste into a Kaggle GPU notebook. This is the real trainer, not train-e2e."""
import os
import subprocess
import sys
from pathlib import Path

os.environ.update({
    "JAGX_SOURCE": "oasst1",
    "JAGX_ROWS": "80000",
    "JAGX_STEPS": "3000",
    "JAGX_SEQ_LEN": "512",
    "JAGX_BATCH_SIZE": "4",
    "JAGX_GRAD_ACCUM": "8",
    "JAGX_WARMUP": "200",
})

ROOT = Path("/kaggle/working/JagX-AI")
if not (ROOT / "pyproject.toml").exists():
    subprocess.check_call([
        "git", "clone", "--depth", "1",
        "https://github.com/Tajudeen001-security/JagX-AI.git",
        str(ROOT),
    ])
os.chdir(ROOT)
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-e", ".[dev]", "datasets", "huggingface_hub"])

import torch
assert torch.cuda.is_available(), "Enable GPU: Settings → Accelerator → GPU, then restart"

subprocess.check_call([
    sys.executable, "scripts/kaggle_train.py",
    "--source", os.environ["JAGX_SOURCE"],
    "--rows", os.environ["JAGX_ROWS"],
    "--steps", os.environ["JAGX_STEPS"],
    "--seq-len", os.environ["JAGX_SEQ_LEN"],
    "--batch-size", os.environ["JAGX_BATCH_SIZE"],
    "--grad-accum", os.environ["JAGX_GRAD_ACCUM"],
    "--vocab-size", "32000",
    "--hidden-size", "512",
    "--layers", "8",
    "--heads", "8",
    "--context-length", "1024",
    "--lr", "3e-4",
    "--resume",
    "--skip-pip",
])
print("Checkpoints:", list((ROOT / "kaggle_checkpoints").glob("*.pt")))
