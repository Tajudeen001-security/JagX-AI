"""African-language data manifest and validation helpers.

This module deliberately separates language *coverage* from language *claims*.
A registry entry is not considered trainable until licensed text/speech data and
community/quality metadata are present. It is safe to run before the large
pretraining job.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REGISTRY = Path(__file__).resolve().parents[1] / "data" / "africa" / "language_registry.json"
REQUIRED = {"name", "code", "family", "regions"}


def load_registry(path: str | Path = REGISTRY) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("target_language_count", 0) < 400:
        raise ValueError("African-language target must remain at least 400")
    languages = data.get("priority_languages", [])
    for item in languages:
        missing = REQUIRED - set(item)
        if missing:
            raise ValueError(f"language entry missing fields: {sorted(missing)}")
    return data


def build_training_manifest(registry: dict[str, Any]) -> list[dict[str, Any]]:
    """Return a manifest consumed by future multilingual data collectors.

    No synthetic data is created here. Each language gets explicit slots for
    text, speech, licences and community validation so low-resource languages
    are not silently replaced by English or high-resource-language data.
    """
    manifest = []
    for lang in registry["priority_languages"]:
        manifest.append({
            "language": lang["name"],
            "code": lang["code"],
            "family": lang["family"],
            "regions": lang["regions"],
            "text": {"sources": [], "verified": False},
            "speech": {"asr_sources": [], "tts_sources": [], "verified": False},
            "local_domains": {
                "agriculture": [],
                "healthcare": [],
                "education": [],
                "infrastructure": [],
                "public_services": []
            },
            "community_validated": False
        })
    return manifest


if __name__ == "__main__":
    registry = load_registry()
    manifest = build_training_manifest(registry)
    print(f"Target languages: {registry['target_language_count']}")
    print(f"Priority languages registered: {len(manifest)}")
    print("Policy: unverified languages are not represented as fully supported.")
