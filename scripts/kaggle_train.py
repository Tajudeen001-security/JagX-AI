#!/usr/bin/env python3
"""Kaggle-ready end-to-end JagX pretraining launcher.

Downloads an approved open corpus, creates deterministic train/validation
splits, trains the native tokenizer, validates packed batches, and runs
resumable CUDA pretraining. Raw data and checkpoints stay outside Git history.
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
    if os.environ.get("JAGX_SKIP_PIP", "0") == "1":
        return
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "datasets", "huggingface_hub"])


def ensure_repo_root() -> Path:
    root = Path(__file__).resolve().parents[1]
    if not (root / "pyproject.toml").is_file():
        raise RuntimeError(f"JagX repository root not found at {root}")
    return root


def fetch_source(root: Path, source: str, rows: int, config: str | None, split: str) -> Path:
    from datasets import load_dataset

    raw = root / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    suffix = f"-{config}" if config else ""
    out = raw / f"{source}{suffix}-{split}-{rows}.jsonl"
    if out.is_file() and out.stat().st_size > 100:
        return out
    repos = {
        "oasst1": ("OpenAssistant/oasst1", "Apache-2.0"),
        "fineweb2": ("HuggingFaceFW/fineweb-2", "ODC-By"),
        "dolma": ("allenai/dolma", "ODC-By"),
    }
    if source not in repos:
        raise ValueError(f"unsupported source: {source}")
    repo, license_name = repos[source]
    kwargs = {"split": split, "streaming": True}
    ds = load_dataset(repo, config, **kwargs) if config else load_dataset(repo, **kwargs)
    count = 0
    with out.open("w", encoding="utf-8") as handle:
        for row in ds:
            text = row.get("text")
            if not text and row.get("messages"):
                text = "\n".join(f"{m.get('role', 'unknown')}: {m.get('content', '')}" for m in row["messages"])
            text = str(text or "").strip()
            if not text:
                continue
            handle.write(json.dumps({
                "id": f"{source}:{count}", "text": text, "source": source,
                "license": license_name, "language": row.get("lang") or row.get("language") or "und",
                "domain": "instruction" if source == "oasst1" else "general", "quality": 1.0, "split": "train",
            }, ensure_ascii=False) + "\n")
            count += 1
            if count >= rows:
                break
    if count == 0:
        raise RuntimeError(f"{source} download returned zero usable records")
    print(f"Downloaded {count:,} real {source} records -> {out}")
    return out


def split_and_make_corpus(source: Path, root: Path, validation_fraction: float) -> tuple[Path, Path, Path]:
    prepared = root / "data" / "prepared"
    prepared.mkdir(parents=True, exist_ok=True)
    train_jsonl, val_jsonl, corpus_txt = prepared / "train.jsonl", prepared / "validation.jsonl", prepared / "tokenizer_corpus.txt"
    train_count = val_count = 0
    seen: set[str] = set()
    with source.open(encoding="utf-8") as src, train_jsonl.open("w", encoding="utf-8") as tr, val_jsonl.open("w", encoding="utf-8") as va, corpus_txt.open("w", encoding="utf-8") as corpus:
        for line in src:
            row = json.loads(line)
            normalized = " ".join(str(row.get("text", "")).split())
            if len(normalized) < 20:
                continue
            key = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
            if key in seen:
                continue
            seen.add(key)
            record = {"text": normalized, "source": row.get("source", source.name), "license": row.get("license", "unknown"), "quality": float(row.get("quality", 1.0))}
            if int(key[:8], 16) / 0xFFFFFFFF < validation_fraction:
                record["split"] = "validation"; va.write(json.dumps(record, ensure_ascii=False) + "\n"); val_count += 1
            else:
                record["split"] = "train"; tr.write(json.dumps(record, ensure_ascii=False) + "\n"); corpus.write(normalized + "\n"); train_count += 1
    if train_count == 0 or val_count == 0:
        raise RuntimeError(f"invalid prepared corpus: train={train_count}, validation={val_count}")
    print(f"Prepared {train_count:,} train + {val_count:,} validation records")
    return train_jsonl, val_jsonl, corpus_txt


def train_tokenizer(root: Path, corpus: Path, vocab_size: int) -> Path:
    from tokenizer.train_tokenizer import train
    out = root / "artifacts" / "kaggle_tokenizer"
    tokenizer = train([str(corpus)], out, vocab_size=vocab_size, min_frequency=2)
    print(f"Tokenizer ready: vocab_size={tokenizer.vocab_size} -> {out}")
    return out


def newest_checkpoint(directory: Path) -> Path | None:
    candidates = list(directory.glob("step-*.pt"))
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def run_training(root: Path, train_jsonl: Path, val_jsonl: Path, tokenizer_path: Path, args: argparse.Namespace) -> dict:
    import torch
    from model import ModelConfig
    from tokenizer import JagXTokenizer
    from training.entrypoint import load_examples, run_training
    from training.pretraining import PretrainingConfig, packed_batches, prepare_examples

    if not torch.cuda.is_available():
        raise RuntimeError("Kaggle GPU is not enabled. In Kaggle: Settings -> Accelerator -> GPU, then restart the session.")
    tokenizer = JagXTokenizer.from_pretrained(tokenizer_path)
    examples = load_examples(train_jsonl)
    examples, stats = prepare_examples(examples, seed=args.seed)
    pre_cfg = PretrainingConfig(seq_len=args.seq_len, batch_size=args.batch_size, max_steps=args.steps, grad_accum=args.grad_accum,
        learning_rate=args.lr, weight_decay=args.weight_decay, warmup_steps=args.warmup_steps, min_lr_ratio=args.min_lr_ratio,
        seed=args.seed, drop_remainder=True)
    try:
        first_batch = next(iter(packed_batches(examples, tokenizer, pre_cfg)))
    except StopIteration as exc:
        raise RuntimeError("Prepared corpus produced zero training batches. Increase rows or reduce --seq-len.") from exc
    print(f"Batch check OK: shape={tuple(first_batch['input_ids'].shape)} tokens={first_batch['input_ids'].numel():,}; accepted={stats.accepted:,}")
    kv_heads = max(1, args.heads // 2)
    model_config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=args.context_length,
        d_model=args.hidden_size,
        n_layers=args.layers,
        n_heads=args.heads,
        n_kv_heads=kv_heads,
        use_sdpa=True,
        gradient_checkpointing=args.hidden_size >= 768,
    )
    out = root / "kaggle_checkpoints"
    resume = newest_checkpoint(out) if args.resume else None
    if resume:
        print(f"Resuming from {resume}")
    result = run_training(train_jsonl, tokenizer_path, model_config, pre_cfg, output_dir=out, resume_from=resume, validation_data_path=val_jsonl, device="cuda")
    result["gpu"] = torch.cuda.get_device_name(0)
    result["cuda_capability"] = list(torch.cuda.get_device_capability(0))
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=("oasst1", "fineweb2", "dolma"), default=os.environ.get("JAGX_SOURCE", "oasst1"))
    parser.add_argument("--config", default=os.environ.get("JAGX_DATASET_CONFIG"))
    parser.add_argument("--split", default=os.environ.get("JAGX_DATASET_SPLIT", "train"))
    parser.add_argument("--rows", type=int, default=int(os.environ.get("JAGX_ROWS", "80000")))
    parser.add_argument("--steps", type=int, default=int(os.environ.get("JAGX_STEPS", "3000")))
    parser.add_argument("--seq-len", type=int, default=int(os.environ.get("JAGX_SEQ_LEN", "512")))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("JAGX_BATCH_SIZE", "4")))
    parser.add_argument("--grad-accum", type=int, default=int(os.environ.get("JAGX_GRAD_ACCUM", "8")))
    parser.add_argument("--vocab-size", type=int, default=int(os.environ.get("JAGX_VOCAB", "32000")))
    parser.add_argument("--context-length", type=int, default=int(os.environ.get("JAGX_CONTEXT", "1024")))
    parser.add_argument("--hidden-size", type=int, default=int(os.environ.get("JAGX_HIDDEN", "512")))
    parser.add_argument("--layers", type=int, default=int(os.environ.get("JAGX_LAYERS", "8")))
    parser.add_argument("--heads", type=int, default=int(os.environ.get("JAGX_HEADS", "8")))
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=int(os.environ.get("JAGX_WARMUP", "100")))
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-pip", action="store_true")
    args = parser.parse_args()
    if args.skip_pip:
        os.environ["JAGX_SKIP_PIP"] = "1"
    install_runtime_deps()
    root = ensure_repo_root()
    source = fetch_source(root, args.source, args.rows, args.config, args.split)
    train_jsonl, val_jsonl, corpus = split_and_make_corpus(source, root, validation_fraction=0.05)
    tokenizer_path = train_tokenizer(root, corpus, args.vocab_size)
    run_training(root, train_jsonl, val_jsonl, tokenizer_path, args)


if __name__ == "__main__":
    main()
