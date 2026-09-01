from __future__ import annotations

import tempfile
from pathlib import Path

from agent.core import JagXAgent
from agent.loop import AgentLoop
from tools.policy import ToolPolicy, PolicyError
from tools.sandbox import WorkspaceSandbox
from tools.registry import ToolResult


def test_sandbox_path_boundary():
    with tempfile.TemporaryDirectory() as d:
        sb = WorkspaceSandbox(d)
        sb.write("a.txt", "hello")
        assert sb.read("a.txt") == "hello"
        try:
            sb.path("../escape.txt")
            assert False, "should escape"
        except ValueError:
            pass


def test_sandbox_command_allowlist():
    with tempfile.TemporaryDirectory() as d:
        sb = WorkspaceSandbox(d)
        # python3 --version is allowlisted; python -c is intentionally blocked by CommandGuard
        result = sb.run_command("python3 --version", timeout_s=10)
        assert "returncode" in result
        assert result.get("ok") is True or result.get("returncode") == 0

        try:
            sb.run_command("rm -rf /")
            assert False, "rm must be blocked"
        except PermissionError:
            pass

        try:
            sb.run_command('python3 -c "print(1)"')
            assert False, "python -c must be blocked"
        except PermissionError:
            pass


def test_agent_policy_and_tool():
    agent = JagXAgent(policy=ToolPolicy(filesystem=True, shell=False))

    def echo(args):
        return ToolResult(ok=True, data=args.get("text", ""))

    agent.register("echo", echo, description="echo", permission="filesystem")
    r = agent.execute_tool("echo", {"text": "hi"})
    assert r.ok and r.data == "hi"

    agent.register("shellish", echo, permission="shell")
    # execute_tool captures PolicyError into ToolResult rather than raising
    r2 = agent.execute_tool("shellish", {})
    assert not r2.ok
    assert r2.error is not None
    assert "denied" in r2.error.lower() or "permission" in r2.error.lower()


def test_agent_loop_retries():
    attempts = {"n": 0}

    def plan():
        return "do"

    def act(a):
        attempts["n"] += 1
        if attempts["n"] < 2:
            raise RuntimeError("transient")
        return "ok"

    def verify(r):
        return r == "ok"

    loop = AgentLoop(max_steps=5, max_retries=2)
    result = loop.run(plan, act, verify)
    assert result == "ok"
    assert attempts["n"] == 2
