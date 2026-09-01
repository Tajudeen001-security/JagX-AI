from dataclasses import asdict, dataclass
import json
from pathlib import Path

@dataclass(frozen=True)
class DatasetRecord:
    name: str
    source: str
    license: str
    domain: str
    language: str
    version: str
    quality_score: float = 0.0
    contamination_checked: bool = False


def write_manifest(records: list[DatasetRecord], path: str):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps([asdict(r) for r in records], indent=2), encoding="utf-8")
