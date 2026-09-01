import tempfile
from pathlib import Path

from games.unreal.generator import generate_unreal_stub
from games.unity.generator import generate_unity_stub


def test_unreal_stub():
    with tempfile.TemporaryDirectory() as d:
        root = generate_unreal_stub(Path(d) / "ue", name="Demo")
        assert (root / "Demo.uproject").exists()
        assert (root / "Content").is_dir()


def test_unity_stub():
    with tempfile.TemporaryDirectory() as d:
        root = generate_unity_stub(Path(d) / "unity")
        assert (root / "Assets").is_dir()
        assert (root / "ProjectSettings" / "ProjectVersion.txt").exists()
