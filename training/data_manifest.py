from __future__ import annotations
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

@dataclass(frozen=True)
class DatasetManifest:
    name: str
    version: str
    source: str
    examples: int
    tokens: int
    sha256: str

    def validate(self) -> "DatasetManifest":
        if not self.name.strip() or not self.version.strip() or not self.source.strip():
            raise ValueError("dataset identity fields are required")
        if self.examples < 0 or self.tokens < 0:
            raise ValueError("dataset counts cannot be negative")
        if len(self.sha256) != 64:
            raise ValueError("sha256 must be a 64-character hex digest")
        int(self.sha256, 16)
        return self

def hash_file(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def save_manifest(manifest: DatasetManifest, path: str) -> None:
    manifest.validate()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(asdict(manifest), indent=2), encoding="utf-8")
    temporary.replace(target)
