import pytest

from runtime.inference_scheduler import InferenceScheduler


def test_scheduler_executes_and_tracks():
    scheduler = InferenceScheduler(lambda x: x * 2)
    try:
        future = scheduler.submit("r1", 21)
        assert future.result(timeout=2) == 42
        assert scheduler.snapshot() == {"active": 0, "completed": 1, "failed": 0}
    finally:
        scheduler.shutdown()


def test_scheduler_rejects_full_queue():
    scheduler = InferenceScheduler(lambda x: x, max_workers=1, max_queue=1)
    try:
        first = scheduler.submit("r1", 1)
        with pytest.raises(RuntimeError, match="queue is full"):
            scheduler.submit("r2", 2)
        assert first.result(timeout=2) == 1
    finally:
        scheduler.shutdown()


def test_scheduler_records_failures():
    scheduler = InferenceScheduler(lambda _: (_ for _ in ()).throw(ValueError("boom")))
    try:
        with pytest.raises(ValueError, match="boom"):
            scheduler.submit("r1", None).result(timeout=2)
        assert scheduler.snapshot()["failed"] == 1
    finally:
        scheduler.shutdown()
