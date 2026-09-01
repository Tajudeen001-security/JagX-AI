from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path


@dataclass(frozen=True)
class CheckpointManifest:
    step: int
    model_config: str
    dataset_manifest: str
    metric: float
    tokens_seen: int

    def validate(self):
        if self.step < 0 or self.tokens_seen < 0:
            raise ValueError("step and tokens_seen must be non-negative")
        if not self.model_config.strip() or not self.dataset_manifest.strip():
            raise ValueError("model and dataset references are required")
        return self


def fingerprint(manifest: CheckpointManifest) -> str:
    manifest.validate()
    payload = json.dumps(manifest.__dict__, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def save(manifest: CheckpointManifest, path: str) -> None:
    manifest.validate()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps({**manifest.__dict__, "fingerprint": fingerprint(manifest)}, indent=2), encoding="utf-8"
    )
