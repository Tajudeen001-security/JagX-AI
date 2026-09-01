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
        # python3 should be allowed
        result = sb.run_command("python3 -c print(1+1)", timeout_s=10)
        # may fail if python flags differ; just ensure it did not raise PermissionError
        assert "returncode" in result
        try:
            sb.run_command("rm -rf /")
            assert False
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
    try:
        agent.execute_tool("shellish", {})
        assert False
    except PolicyError:
        pass


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
