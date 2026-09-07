from pathlib import Path

import pytest

from agent.computer_control import AccessGrant, AccessLevel, ComputerAction, ComputerController, ComputerRequest


class FakeAdapter:
    def execute(self, request):
        return {"ok": True, "action": request.action.value}


def test_read_only_blocks_cursor_write():
    grant = AccessGrant(AccessLevel.READ_ONLY)
    controller = ComputerController(FakeAdapter(), grant)
    with pytest.raises(PermissionError):
        controller.execute(ComputerRequest(ComputerAction.CLICK, target="screen"))


def test_workspace_contains_file(tmp_path: Path):
    grant = AccessGrant(AccessLevel.WORKSPACE, workspace=tmp_path)
    controller = ComputerController(FakeAdapter(), grant)
    result = controller.execute(ComputerRequest(ComputerAction.WRITE_FILE, target=str(tmp_path / "a.txt")))
    assert result["ok"]


def test_destructive_action_requires_confirmation(tmp_path: Path):
    grant = AccessGrant(AccessLevel.FULL_CONTROL)
    controller = ComputerController(FakeAdapter(), grant)
    result = controller.execute(ComputerRequest(ComputerAction.DELETE_FILE, target=str(tmp_path / "a.txt")))
    assert result["needs_confirmation"] is True
