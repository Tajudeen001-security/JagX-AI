import tempfile
from pathlib import Path

from games.godot.generator import generate_godot_project


def test_generate_godot_project():
    with tempfile.TemporaryDirectory() as d:
        root = generate_godot_project(Path(d) / "game", name="TestGame")
        assert (root / "project.godot").exists()
        assert (root / "main.tscn").exists()
        assert (root / "main.gd").exists()
        content = (root / "project.godot").read_text(encoding="utf-8")
        assert "TestGame" in content
