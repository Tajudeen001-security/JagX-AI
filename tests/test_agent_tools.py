from agent.plan import AgentPlan, AgentStep
from agent.tool_registry import ToolRegistry, ToolSpec
from agent.tools import WorkspaceTool


def test_registry_requires_confirmation():
    registry = ToolRegistry()
    registry.register(ToolSpec("write", "write a file", lambda **kw: kw, requires_confirmation=True))
    try:
        registry.call("write", path="x")
        raise AssertionError("confirmation should be required")
    except PermissionError:
        pass


def test_workspace_blocks_escape(tmp_path):
    tool = WorkspaceTool(tmp_path)
    tool.write_file("ok.txt", "hello")
    assert tool.read_file("ok.txt") == "hello"
    try:
        tool.read_file("../outside.txt")
        raise AssertionError("path traversal should be blocked")
    except PermissionError:
        pass


def test_plan_validation():
    plan = AgentPlan("build and test", [AgentStep("tool", tool="write")])
    assert plan.validate().to_dict()["goal"] == "build and test"
