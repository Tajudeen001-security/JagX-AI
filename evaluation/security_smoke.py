from __future__ import annotations
from security.command_guard import CommandGuard
from security.policy import Capability, SecurityPolicy

def run_security_smoke():
    guard=CommandGuard()
    assert guard.validate('python --version')[0]=='python'
    for command in ('rm -rf /','python -c "print(1)"','python test.py && rm x','python test.py > out'):
        try: guard.validate(command)
        except (PermissionError,ValueError): pass
        else: raise AssertionError(f'unsafe command accepted: {command}')
    policy=SecurityPolicy()
    assert policy.check(Capability.READ_FILES)
    assert not policy.check(Capability.NETWORK)
    assert not policy.check(Capability.FINANCIAL)
    return True

if __name__=='__main__': print('security smoke: PASS' if run_security_smoke() else 'FAIL')
