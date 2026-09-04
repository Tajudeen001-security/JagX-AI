from __future__ import annotations

import tempfile
from pathlib import Path

from runtime.orchestrator import RequestStatus, TaskKind, build_default_orchestrator


def test_code_handler_write_and_test_in_temp_workspace():
    orch = build_default_orchestrator()
    with tempfile.TemporaryDirectory() as td:
        files = {
            "hello.py": "def add(a, b):\n    return a + b\n",
            "test_hello.py": "from hello import add\n\ndef test_add():\n    assert add(2, 3) == 5\n",
        }
        result = orch.execute(
            {
                "kind": "code",
                "workspace": td,
                "files": files,
                "test_command": "python3 -m pytest -q",
                "timeout_s": 60,
            }
        )
        assert result.status == RequestStatus.SUCCEEDED
        assert result.kind == TaskKind.CODE
        data = result.data or {}
        # May succeed if pytest available; still must return structured payload
        assert "request_id" in data or "backend" in data or "status" in data
        if data.get("ok") is True:
            assert (Path(td) / "hello.py").exists()
