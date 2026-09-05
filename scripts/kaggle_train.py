#!/usr/bin/env python3
"""Kaggle-ready end-to-end JagX pretraining launcher.

Downloads a real open corpus, creates deterministic train/validation splits,
trains a tokenizer from the downloaded corpus, verifies non-empty batches, and
runs native JagX causal-LM training on CUDA. Large raw data and checkpoints are
kept outside Git history when this is run on Kaggle.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def install_runtime_deps() -> None:
    """Install only the Kaggle-side dataset dependency when requested."""
    if os.environ.get("JAGX_SKIP_PIP", "0") == "1":
        return
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "datasets", "huggingface_hub"])


def ensure_repo_root() -> Path:
    root = Path(__file__).resolve().parents[1]
    if not (root / "pyproject.toml").is_file():
        raise RuntimeError(f"JagX repository root not found at {root}")
    return root


def fetch_oasst(root: Path, rows: int) -> Path:
    from datasets import load_dataset

    raw = root / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    out = raw / f"oasst1-kaggle-{rows}.jsonl"
    if out.is_file() and out.stat().st_size > 100:
        return out

    ds = load_dataset("OpenAssistant/oasst1", split="train", streaming=True)
    count = 0
    with out.open("w", encoding="utf-8") as handle:
        for row in ds:
            text = str(row.get("text") or "").strip()
            if not text:
                continue
            handle.write(json.dumps({
                "id": f"oasst1:{count}",
                "text": text,
                "source": "OpenAssistant/oasst1",
                "license": "Apache-2.0",
                "language": row.get("lang") or "und",
                "domain": "instruction",
                "quality": 1.0,
                "split": "train",
            }, ensure_ascii=False) + "\n")
            count += 1
            if count >= rows:
                break
    if count == 0:
        raise RuntimeError("OASST1 download returned zero usable records")
    print(f"Downloaded {count:,} real OASST1 records -> {out}")
    return out


def split_and_make_corpus(source: Path, root: Path, validation_fraction: float, seed: int) -> tuple[Path, Path, Path]:
    prepared = root / "data" / "prepared"
    prepared.mkdir(parents=True, exist_ok=True)
    train_jsonl = prepared / "train.jsonl"
    val_jsonl = prepared / "validation.jsonl"
    corpus_txt = prepared / "tokenizer_corpus.txt"

    train_count = val_count = 0
    seen: set[str] = set()
    with source.open(encoding="utf-8") as src, train_jsonl.open("w", encoding="utf-8") as tr, val_jsonl.open("w", encoding="utf-8") as va, corpus_txt.open("w", encoding="utf-8") as corpus:
        for line in src:
            row = json.loads(line)
            text = str(row.get("text", "")).strip()
            normalized = " ".join(text.split())
            if len(normalized) < 20:
                continue
            key = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
            if key in seen:
                continue
            seen.add(key)
            record = {
                "text": normalized,
                "source": row.get("source", "OpenAssistant/oasst1"),
                "license": row.get("license", "Apache-2.0"),
                "quality": float(row.get("quality", 1.0)),
            }
            bucket = int(key[:8], 16) / 0xFFFFFFFF
            if bucket < validation_fraction:
                record["split"] = "validation"
                va.write(json.dumps(record, ensure_ascii=False) + "\n")
                val_count += 1
            else:
                record["split"] = "train"
                tr.write(json.dumps(record, ensure_ascii=False) + "\n")
                corpus.write(normalized + "\n")
                train_count += 1

    if train_count == 0:
        raise RuntimeError("Prepared training corpus is empty")
    if val_count == 0:
        raise RuntimeError("Prepared validation corpus is empty")
    print(f"Prepared {train_count:,} train + {val_count:,} validation records")
    return train_jsonl, val_jsonl, corpus_txt


def train_tokenizer(root: Path, corpus: Path, vocab_size: int) -> Path:
    from tokenizer.train_tokenizer import train

    out = root / "artifacts" / "kaggle_tokenizer"
    tokenizer = train([str(corpus)], out, vocab_size=vocab_size, min_frequency=2)
    print(f"Tokenizer ready: vocab_size={tokenizer.vocab_size} -> {out}")
    return out


def run_training(root: Path, train_jsonl: Path, val_jsonl: Path, tokenizer_path: Path, args: argparse.Namespace) -> dict:
    import torch
    from model import ModelConfig
    from training.entrypoint import run_training
    from training.pretraining import PretrainingConfig

    if not torch.cuda.is_available():
        raise RuntimeError("Kaggle GPU is not enabled. In Kaggle: Settings -> Accelerator -> GPU, then restart the session.")

    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    model_config = ModelConfig(
        vocab_size=__import__("tokenizer").JagXTokenizer.from_pretrained(tokenizer_path).vocab_size,
        max_seq_len=args.context_length,
        d_model=args.hidden_size,
        n_layers=args.layers,
        n_heads=args.heads,
    )
    pre_cfg = PretrainingConfig(
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        max_steps=args.steps,
        grad_accum=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        drop_remainder=True,
    )

    # Fail before allocating the model if packing cannot produce a batch.
    from training.entrypoint import load_examples
    from training.pretraining import packed_batches, prepare_examples
    tokenizer = __import__("tokenizer").JagXTokenizer.from_pretrained(tokenizer_path)
    examples = load_examples(train_jsonl)
    examples, stats = prepare_examples(examples, seed=args.seed)
    batches = packed_batches(examples, tokenizer, pre_cfg)
    try:
        first_batch = next(iter(batches))
    except StopIteration as exc:
        raise RuntimeError("Prepared corpus produced zero training batches. Increase rows or reduce --seq-len.") from exc
    print(f"Batch check OK: shape={tuple(first_batch['input_ids'].shape)} tokens={first_batch['input_ids'].numel():,}; accepted={stats.accepted:,}")

    out = root / "kaggle_checkpoints"
    result = run_training(
        train_jsonl,
        tokenizer_path,
        model_config,
        pre_cfg,
        output_dir=out,
        validation_data_path=val_jsonl,
        device="cuda",
    )
    result["device"] = str(device)
    result["gpu"] = torch.cuda.get_device_name(0)
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=int(os.environ.get("JAGX_ROWS", "50000")))
    parser.add_argument("--steps", type=int, default=int(os.environ.get("JAGX_STEPS", "1000")))
    parser.add_argument("--seq-len", type=int, default=int(os.environ.get("JAGX_SEQ_LEN", "512")))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("JAGX_BATCH_SIZE", "4")))
    parser.add_argument("--grad-accum", type=int, default=int(os.environ.get("JAGX_GRAD_ACCUM", "8")))
    parser.add_argument("--vocab-size", type=int, default=32768)
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--hidden-size", type=int, default=384)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-pip", action="store_true")
    args = parser.parse_args()

    if args.skip_pip:
        os.environ["JAGX_SKIP_PIP"] = "1"
    install_runtime_deps()
    root = ensure_repo_root()
    source = fetch_oasst(root, args.rows)
    train_jsonl, val_jsonl, corpus = split_and_make_corpus(source, root, validation_fraction=0.05, seed=args.seed)
    tokenizer_path = train_tokenizer(root, corpus, args.vocab_size)
    run_training(root, train_jsonl, val_jsonl, tokenizer_path, args)


if __name__ == "__main__":
    main()
