from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PreferenceExample:
    prompt: str
    chosen: str
    rejected: str
    source: str = "unknown"

    def validate(self) -> "PreferenceExample":
        if not self.prompt.strip() or not self.chosen.strip() or not self.rejected.strip():
            raise ValueError("prompt, chosen and rejected must be non-empty")
        if self.chosen.strip() == self.rejected.strip():
            raise ValueError("chosen and rejected responses must differ")
        return self


def load_jsonl(path: str | Path) -> list[PreferenceExample]:
    rows: list[PreferenceExample] = []
    with Path(path).open(encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {n}") from exc
            rows.append(PreferenceExample(
                prompt=str(row.get("prompt", "")),
                chosen=str(row.get("chosen", "")),
                rejected=str(row.get("rejected", "")),
                source=str(row.get("source", "unknown")),
            ).validate())
    if not rows:
        raise ValueError("preference dataset is empty")
    return rows
