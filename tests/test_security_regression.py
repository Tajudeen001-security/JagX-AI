from security.regression import all_passed, run_security_regression


def test_security_regression_suite():
    cases = run_security_regression()
    assert cases, "expected regression cases"
    failed = [c for c in cases if not c.passed]
    assert not failed, f"failed: {failed}"
    assert all_passed(cases)
