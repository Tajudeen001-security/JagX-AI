"""End-to-end local path: data → tokenizer → train → checkpoint → generate.

Uses only local files and the native JagX stack. No external AI provider.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import torch

from inference.loader import generate_text, load_model, load_tokenizer
from model import JagXTransformer, ModelConfig
from tokenizer.train_tokenizer import train as train_tokenizer
from training.entrypoint import load_examples, run_training
from training.pretraining import PretrainingConfig


DEFAULT_CORPUS = """The agent plans tasks as a directed graph.
Sandbox tools enforce path boundaries and command allowlists.
Training saves checkpoints so long runs can resume later.
JagX keeps inference local when a checkpoint is available.
Memory stores episodic notes and retrieves them by lexical overlap.
Coding workflows write files then execute tests inside the workspace.
Reasoning improves when models are trained on clean open data.
Vision capabilities require multimodal weights or a cloud backend.
Paper trading never moves real money without explicit authorization.
Evaluation gates block false claims of frontier performance.
"""


def write_bootstrap_data(out_dir: Path) -> tuple[Path, Path]:
    """Write a tiny open-style corpus + JSONL for training."""
    out_dir.mkdir(parents=True, exist_ok=True)
    text_path = out_dir / "corpus.txt"
    jsonl_path = out_dir / "train.jsonl"
    # Repeat to give BPE and LM enough tokens on tiny runs
    body = (DEFAULT_CORPUS + "\n") * 40
    text_path.write_text(body, encoding="utf-8")
    with jsonl_path.open("w", encoding="utf-8") as f:
        for line in DEFAULT_CORPUS.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            for _ in range(8):
                f.write(
                    json.dumps(
                        {
                            "text": line,
                            "source": "bootstrap",
                            "split": "train",
                            "license": "project-internal",
                            "quality": 1.0,
                        }
                    )
                    + "\n"
                )
    return text_path, jsonl_path


def run_e2e(
    work_dir: str | Path,
    *,
    steps: int = 30,
    vocab_size: int = 512,
    device: Optional[str] = None,
    prompt: str = "The agent plans",
) -> dict[str, Any]:
    """Train a tiny model and run one real generation."""
    root = Path(work_dir)
    data_dir = root / "data"
    tok_dir = root / "tokenizer"
    ckpt_dir = root / "checkpoints"
    text_path, jsonl_path = write_bootstrap_data(data_dir)

    tokenizer = train_tokenizer(
        [str(text_path)],
        tok_dir,
        vocab_size=vocab_size,
        min_frequency=1,
    )
    # Align model vocab to trained tokenizer
    model_config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=64,
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        d_ff=128,
        dropout=0.0,
        use_swiglu=True,
        use_rms_norm=True,
        tie_embeddings=True,
    )
    pretraining = PretrainingConfig(
        seq_len=32,
        batch_size=2,
        max_steps=steps,
        grad_accum=1,
        learning_rate=3e-3,
        weight_decay=0.0,
        seed=42,
    )
    result = run_training(
        jsonl_path,
        tok_dir,
        model_config,
        pretraining,
        output_dir=ckpt_dir,
        device=device,
    )

    # Prefer final checkpoint written by trainer
    candidates = sorted(ckpt_dir.glob("step-*.pt"))
    if not candidates:
        raise FileNotFoundError(f"no checkpoint written under {ckpt_dir}")
    ckpt_path = candidates[-1]

    model, _ = load_model(ckpt_path, device=device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tok = load_tokenizer(tok_dir)
    text = generate_text(
        model,
        tok,
        prompt,
        max_new_tokens=16,
        temperature=0.8,
        top_k=20,
        seed=0,
    )
    return {
        "tokenizer_dir": str(tok_dir),
        "checkpoint": str(ckpt_path),
        "train": result,
        "prompt": prompt,
        "generated": text,
        "backend": "local-jagx",
        "external_ai_api_required": False,
    }


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="JagX tiny train→generate e2e")
    p.add_argument("--work-dir", default="artifacts/e2e_tiny")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--vocab-size", type=int, default=512)
    p.add_argument("--device", default=None)
    p.add_argument("--prompt", default="The agent plans")
    args = p.parse_args()
    out = run_e2e(
        args.work_dir,
        steps=args.steps,
        vocab_size=args.vocab_size,
        device=args.device,
        prompt=args.prompt,
    )
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
