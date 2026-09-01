from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GameProject:
    engine: str
    name: str
    root: Path


class GameEngineAdapter:
    """Engine-neutral project builder used by JagX coding agents."""

    engine = "generic"

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()

    def create(self, name: str, files: dict[str, str]) -> GameProject:
        project = self.root / name
        for relative, content in files.items():
            path = (project / relative).resolve()
            if project not in path.parents and path != project:
                raise ValueError("game file escapes project")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        return GameProject(self.engine, name, project)


class GodotAdapter(GameEngineAdapter):
    engine = "godot"


class UnityAdapter(GameEngineAdapter):
    engine = "unity"

    def create(self, name: str, files: dict[str, str]) -> GameProject:
        files = {"ProjectSettings/ProjectVersion.txt": "m_EditorVersion: 2022.3.0f1\n", **files}
        return super().create(name, files)


class UnrealAdapter(GameEngineAdapter):
    engine = "unreal"

    def create(self, name: str, files: dict[str, str]) -> GameProject:
        files = {f"{name}.uproject": '{"FileVersion": 3, "EngineAssociation": "5.4"}\n', **files}
        return super().create(name, files)


def adapter_for(engine: str, root: str | Path) -> GameEngineAdapter:
    adapters = {"godot": GodotAdapter, "unity": UnityAdapter, "unreal": UnrealAdapter}
    try:
        return adapters[engine.lower()](root)
    except KeyError as exc:
        raise ValueError(f"unsupported engine: {engine}") from exc
