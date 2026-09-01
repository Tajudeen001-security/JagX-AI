from __future__ import annotations

import tempfile
from pathlib import Path

from coding import CodingEngine
from tools.sandbox import WorkspaceSandbox


def test_write_and_test_pass():
    with tempfile.TemporaryDirectory() as d:
        sb = WorkspaceSandbox(d)
        eng = CodingEngine(sb)
        files = {
            "m.py": "def add(a, b):\n    return a + b\n",
            "test_m.py": "from m import add\n\ndef test_add():\n    assert add(1, 2) == 3\n",
        }
        result = eng.write_and_test(files, test_command="python3 -m pytest -q test_m.py")
        assert result.ok, result.test_output
        assert "m.py" in result.files_written


def test_repair_loop():
    with tempfile.TemporaryDirectory() as d:
        sb = WorkspaceSandbox(d)
        eng = CodingEngine(sb, max_repair_attempts=3)

        broken = {
            "m.py": "def add(a, b):\n    return a - b\n",
            "test_m.py": "from m import add\n\ndef test_add():\n    assert add(1, 2) == 3\n",
        }

        def repair(files, output):
            files = dict(files)
            files["m.py"] = "def add(a, b):\n    return a + b\n"
            return files

        result = eng.repair_loop(broken, repair, test_command="python3 -m pytest -q test_m.py")
        assert result.ok, result.test_output
        assert result.attempts >= 1
