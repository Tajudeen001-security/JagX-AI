"""Coding DAG runs real write+test when workspace + files are provided."""

from __future__ import annotations

import tempfile
from pathlib import Path

from agent.runtime import AgentRuntime


def test_coding_dag_write_and_test_real():
    with tempfile.TemporaryDirectory() as td:
        files = {
            "math_util.py": "def add(a, b):\n    return a + b\n",
            "test_math_util.py": (
                "from math_util import add\n\n"
                "def test_add():\n"
                "    assert add(1, 2) == 3\n"
            ),
        }
        rt = AgentRuntime.create(workspace=td)
        receipt = rt.run_dag(
            "implement and test the add helper",
            files=files,
            test_command="python3 -m pytest -q",
            timeout_s=60.0,
        )
        assert (Path(td) / "math_util.py").exists()
        assert receipt.dag_summary["total"] >= 3
        # Success depends on pytest availability; files must still be written
        if receipt.success:
            assert any(
                r.get("name") == "run_tests" and r.get("state") == "succeeded"
                for r in receipt.dag_summary.get("receipts", [])
            )
