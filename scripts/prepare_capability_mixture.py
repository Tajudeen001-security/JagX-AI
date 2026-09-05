#!/usr/bin/env python3
"""Build a provenance-preserving capability corpus from open datasets.

This is deliberately a preparation/SFT corpus builder, not a claim of a
frontier foundation corpus. It streams upstream datasets, normalizes common
schemas, filters short/duplicate examples, and writes JSONL without putting raw
datasets in Git.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

SOURCES = [
    ("oasst1", "OpenAssistant/oasst1", None, "train", 50_000, "instruction_conversation", "Apache-2.0"),
    ("openr1_math", "open-r1/OpenR1-Math-220k", "default", "train", 75_000, "mathematics_reasoning", "Apache-2.0"),
    ("codefeedback_python", "fxmeng/CodeFeedback-Python105K", None, "train", 75_000, "programming_code_execution", "Apache-2.0"),
    ("helpful_instructions", "HuggingFaceH4/helpful-instructions", None, "train", 25_000, "instruction_following", "Apache-2.0"),
    ("swe_coding", "obaydata/swe-coding-instruction-following", None, "train", 5_000, "software_engineering", "Apache-2.0"),
]


def text(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v.strip()
    if isinstance(v, list):
        parts = []
        for item in v:
            if isinstance(item, dict):
                role = item.get("role", "")
                content = item.get("content", item.get("text", ""))
                if content:
                    parts.append(f"{role}: {content}" if role else str(content))
            elif item:
                parts.append(str(item))
        return "\n".join(parts).strip()
    return str(v).strip()


def normalize(source_id: str, row: dict[str, Any]) -> str:
    # Math reasoning: retain problem, verified solution trace and answer.
    if source_id == "openr1_math":
        problem = text(row.get("problem"))
        solution = text(row.get("solution"))
        answer = text(row.get("answer"))
        if problem and solution:
            return f"Problem:\n{problem}\n\nSolution:\n{solution}\n\nAnswer:\n{answer}".strip()

    # Common instruction/code schemas.
    instruction = text(row.get("instruction"))
    inp = text(row.get("input"))
    output = text(row.get("output", row.get("response", row.get("answer"))))
    if instruction and output:
        extra = f"\nInput:\n{inp}" if inp else ""
        return f"Instruction:\n{instruction}{extra}\n\nResponse:\n{output}".strip()

    # OASST and generic conversational datasets.
    if row.get("messages"):
        return text(row["messages"])
    if row.get("text"):
        return text(row["text"])
    if row.get("prompt") and row.get("completion"):
        return f"Prompt:\n{text(row['prompt'])}\n\nResponse:\n{text(row['completion'])}"
    return ""


def stream_dataset(dataset: str, config: str | None, split: str):
    from datasets import load_dataset
    kwargs = {"split": split, "streaming": True}
    if config:
        return load_dataset(dataset, config, **kwargs)
    return load_dataset(dataset, **kwargs)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="data/prepared/capability_mixture.jsonl")
    p.add_argument("--scale", type=float, default=1.0, help="Scale all source caps; 0.1 is a small Kaggle smoke test")
    p.add_argument("--min-chars", type=int, default=40)
    args = p.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    totals: dict[str, int] = {}

    with out.open("w", encoding="utf-8") as handle:
        for source_id, dataset, config, split, cap, role, license_name in SOURCES:
            target = max(1, int(cap * args.scale))
            count = 0
            print(f"Streaming {dataset} [{role}] target={target:,}")
            ds = stream_dataset(dataset, config, split)
            for row in ds:
                normalized = normalize(source_id, dict(row))
                normalized = " ".join(normalized.split())
                if len(normalized) < args.min_chars:
                    continue
                digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
                if digest in seen:
                    continue
                seen.add(digest)
                handle.write(json.dumps({
                    "id": f"{source_id}:{digest[:16]}",
                    "text": normalized,
                    "source": dataset,
                    "source_id": source_id,
                    "license": license_name,
                    "role": role,
                    "quality": 1.0,
                    "split": "train",
                }, ensure_ascii=False) + "\n")
                count += 1
                if count >= target:
                    break
            totals[source_id] = count
            print(f"  accepted={count:,}")
            if count == 0:
                raise RuntimeError(f"No usable records were produced from {dataset}")

    print("\nCapability mixture complete")
    print(json.dumps({"sources": totals, "total": sum(totals.values()), "output": str(out)}, indent=2))


if __name__ == "__main__":
    main()
