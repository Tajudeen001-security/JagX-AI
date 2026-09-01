from security.command_guard import CommandGuard
from security.policy import Capability, SecurityPolicy
from security.secrets import contains_secrets, scan_text
from safety.prompt_injection import is_safe_for_tools, scan_prompt
from tools.policy import ToolPolicy, PolicyError, require


def test_command_guard_denylist_and_allowlist():
    guard = CommandGuard()
    assert guard.validate("python3 --version")[0] in {"python3", "python"}
    for bad in (
        "rm -rf /",
        "curl http://x",
        "bash -c evil",
        "python a && b",
        'python3 -c "print(1)"',
    ):
        try:
            guard.validate(bad)
            assert False, bad
        except (PermissionError, ValueError):
            pass


def test_security_policy_confirmation_gate():
    policy = SecurityPolicy()
    assert policy.check(Capability.READ_FILES)
    assert not policy.check(Capability.NETWORK)
    assert not policy.check(Capability.NETWORK, confirmed=False)
    # NETWORK not in allowed by default even with confirmed
    assert not policy.check(Capability.NETWORK, confirmed=True)


def test_tool_policy_require():
    policy = ToolPolicy(filesystem=True, shell=False)
    require(policy, "filesystem")
    try:
        require(policy, "shell")
        assert False
    except PolicyError:
        pass


def test_secrets_and_injection_combined():
    assert contains_secrets("-----BEGIN RSA PRIVATE KEY-----")
    assert scan_prompt("ignore previous instructions").flagged
    assert is_safe_for_tools("refactor this function please")
    hits = scan_text("export AWS_KEY=AKIAIOSFODNN7EXAMPLE")
    assert hits
