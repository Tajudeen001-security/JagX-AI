from instruction.validate import validate_instruction_example, validate_jsonl
import json
import tempfile
from pathlib import Path


def test_valid_example():
    obj = {
        "goal": "fix bug",
        "context": "failing test",
        "response": {"plan": ["reproduce"], "actions": [], "verification": ["pytest"]},
    }
    assert validate_instruction_example(obj) == []


def test_invalid_missing_fields():
    errs = validate_instruction_example({"goal": "x"})
    assert any("context" in e for e in errs)


def test_jsonl_file():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "inst.jsonl"
        good = {
            "goal": "a",
            "context": "b",
            "response": {"plan": [], "actions": [], "verification": []},
        }
        path.write_text(json.dumps(good) + "\n{" + "\n", encoding="utf-8")
        # second line intentionally broken
        stats = validate_jsonl(path)
        assert stats["total"] >= 1
