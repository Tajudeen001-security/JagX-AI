#!/usr/bin/env python3
"""Reproducibly fetch approved training sources for JagX-AI.

The repository intentionally stores manifests and processing code rather than
multi-gigabyte raw corpora. This script downloads selected upstream datasets
into data/raw/ and writes a provenance manifest for every fetched source.

Examples:
  python scripts/download_training_data.py --source oasst1 --max-rows 50000
  python scripts/download_training_data.py --source oasst1 --max-rows 10000
  python scripts/download_training_data.py --source fineweb2 --config eng_Latn --split train --stream

Requires: pip install datasets huggingface_hub
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
META = ROOT / "data" / "manifests" / "downloaded_sources.jsonl"

SOURCES = {
    "oasst1": {
        "repo": "OpenAssistant/oasst1",
        "license": "Apache-2.0",
        "default_split": "train",
    },
    "fineweb2": {
        "repo": "HuggingFaceFW/fineweb-2",
        "license": "ODC-By",
        "default_split": "train",
    },
    "dolma": {
        "repo": "allenai/dolma",
        "license": "ODC-By",
        "default_split": "train",
    },
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, choices=sorted(SOURCES))
    p.add_argument("--config", default=None)
    p.add_argument("--split", default=None)
    p.add_argument("--max-rows", type=int, default=10000)
    p.add_argument("--stream", action="store_true")
    args = p.parse_args()

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Install dependencies first: pip install datasets huggingface_hub") from exc

    info = SOURCES[args.source]
    split = args.split or info["default_split"]
    RAW.mkdir(parents=True, exist_ok=True)
    META.parent.mkdir(parents=True, exist_ok=True)

    kwargs = {"split": split, "streaming": args.stream}
    if args.config:
        ds = load_dataset(info["repo"], args.config, **kwargs)
    else:
        ds = load_dataset(info["repo"], **kwargs)

    out = RAW / f"{args.source}-{split}.jsonl"
    count = 0
    with out.open("w", encoding="utf-8") as f:
        for row in ds:
            # Normalize common dataset schemas into JagX's canonical text field.
            text = row.get("text")
            if not text and row.get("messages"):
                text = "\n".join(
                    f"{m.get('role', 'unknown')}: {m.get('content', '')}"
                    for m in row["messages"]
                )
            if not text:
                continue
            record = {
                "id": f"{args.source}:{count}",
                "text": str(text),
                "source": args.source,
                "license": info["license"],
                "language": row.get("lang") or row.get("language") or "und",
                "domain": "instruction" if args.source == "oasst1" else "general",
                "quality": 1.0,
                "split": split,
                "contamination_status": "unknown",
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
            if count >= args.max_rows:
                break

    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source": args.source,
        "repo": info["repo"],
        "license": info["license"],
        "split": split,
        "config": args.config,
        "rows": count,
        "path": str(out.relative_to(ROOT)),
    }
    with META.open("a", encoding="utf-8") as f:
        f.write(json.dumps(manifest, ensure_ascii=False) + "\n")
    print(f"Fetched {count:,} records -> {out}")


if __name__ == "__main__":
    main()
