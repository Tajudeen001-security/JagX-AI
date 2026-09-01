from __future__ import annotations

from pathlib import Path


def generate_unreal_stub(root: str | Path, *, name: str = "JagXUnrealProject") -> Path:
    """Create a minimal Unreal-style directory layout (stub, not a full .uproject binary)."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{name}.uproject").write_text(
        '{\n\t"FileVersion": 3,\n\t"EngineAssociation": "5.4",\n\t"Category": "",\n\t"Description": "JagX generated Unreal stub",\n\t"Modules": []\n}\n',
        encoding="utf-8",
    )
    (root / "Source").mkdir(exist_ok=True)
    (root / "Content").mkdir(exist_ok=True)
    (root / "README_JAGX.md").write_text(
        "# JagX Unreal stub\n\nThis is a structural scaffold only. Full Unreal project generation requires the Unreal toolchain on the host.\n",
        encoding="utf-8",
    )
    return root
