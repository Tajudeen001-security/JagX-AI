from __future__ import annotations

from tools.registry import ToolRegistry


def test_default_tools_echo_and_list():
    reg = ToolRegistry.create_default()
    names = {t.name for t in reg.list_tools()}
    assert "echo" in names
    assert "now" in names
    assert "list_tools" in names

    r = reg.run("echo", {"message": "hello"})
    assert r.ok
    assert r.data["echo"] == "hello"

    listed = reg.run("list_tools", {})
    assert listed.ok
    assert len(listed.data["tools"]) >= 3
