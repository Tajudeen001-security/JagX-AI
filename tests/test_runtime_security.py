from security.command_guard import CommandGuard
from security.policy import Capability, SecurityPolicy
from runtime.request_limits import RequestLimits
from runtime.cache import BoundedCache


def test_command_guard_blocks_composition():
    guard = CommandGuard()
    assert guard.validate("python --version")[0] == "python"
    for command in (
        "python x && echo bad",
        "python x > out",
        "python x | cat",
        'python -c "print(1)"',
    ):
        try:
            guard.validate(command)
        except PermissionError:
            pass
        else:
            raise AssertionError(command)


def test_policy_denies_high_risk_by_default():
    policy = SecurityPolicy()
    assert policy.check(Capability.READ_FILES)
    assert not policy.check(Capability.NETWORK)
    assert not policy.check(Capability.FINANCIAL)


def test_limits():
    limits = RequestLimits()
    limits.validate_text(100, 100)
    limits.validate_video(30)
    for fn in (lambda: limits.validate_text(999999, 1), lambda: limits.validate_video(999999)):
        try:
            fn()
        except ValueError:
            pass
        else:
            raise AssertionError("limit was not enforced")


def test_cache_is_bounded():
    cache = BoundedCache(2)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("c", 3)
    assert len(cache) == 2
    assert cache.get("a") is None
    assert cache.get("c") == 3
