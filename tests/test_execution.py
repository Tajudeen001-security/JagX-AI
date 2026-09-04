from runtime.execution import ExecutionEngine


def test_execution_success():
    result = ExecutionEngine().execute("x", lambda: 42)
    assert result.status == "succeeded"
    assert result.result == 42
    assert result.attempts == 1


def test_execution_retry_and_failure():
    calls = []
    def fail():
        calls.append(1)
        raise RuntimeError("boom")
    result = ExecutionEngine().execute("x", fail, retries=2)
    assert result.status == "failed"
    assert result.attempts == 3
    assert len(calls) == 3
