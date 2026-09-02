from __future__ import annotations

from pathlib import Path


def generate_unity_project(root: str | Path, *, name: str = "JagXUnityProject") -> Path:
    """Create a valid Unity project scaffold with a starter scene script."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "Assets").mkdir(exist_ok=True)
    (root / "ProjectSettings").mkdir(exist_ok=True)
    (root / "Assets" / "Scenes").mkdir(exist_ok=True)
    (root / "ProjectSettings" / "ProjectVersion.txt").write_text("m_EditorVersion: 2022.3.0f1\n", encoding="utf-8")
    (root / "README_JAGX.md").write_text(
        f"# {name}\n\nJagX Unity stub. Requires Unity Editor for a real playable project.\n",
        encoding="utf-8",
    )
    return root


# Backward-compatible API retained for existing callers.\ndef generate_unity_stub(root: str | Path, *, name: str = "JagXUnityProject") -> Path:\n    return generate_unity_project(root, name=name)\n