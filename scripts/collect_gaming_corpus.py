#!/usr/bin/env python3
"""Build a licensed gaming-instruction JSONL for JagX capability training.

Always includes the in-repo MIT seed set. Optionally appends extra JSONL files
the user already downloaded under an open license.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = str(row.get("text", "")).strip()
            if len(text) < 40:
                continue
            rows.append({
                "text": text,
                "source": row.get("source", path.name),
                "license": row.get("license", "unknown"),
                "domain": row.get("domain", "games"),
                "quality": float(row.get("quality", 1.0)),
                "split": row.get("split", "train"),
            })
    return rows


def dedupe(rows: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for row in rows:
        key = hashlib.sha256(" ".join(row["text"].split()).encode("utf-8")).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default="data/seed/gaming_instructions.jsonl")
    parser.add_argument("--extra", action="append", default=[], help="Additional JSONL paths")
    parser.add_argument("--out", default="data/prepared/gaming_train.jsonl")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    rows = load_jsonl(root / args.seed)
    for extra in args.extra:
        rows.extend(load_jsonl(Path(extra)))
    rows = dedupe(rows)
    out = Path(args.out)
    if not out.is_absolute():
        out = root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} gaming records -> {out}")


if __name__ == "__main__":
    main()
