from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REQUIRED_TOP = ("goal", "context", "response")
REQUIRED_RESPONSE = ("plan", "actions", "verification")


def validate_instruction_example(obj: dict[str, Any]) -> list[str]:
    """Return a list of validation errors (empty means valid)."""
    errors: list[str] = []
    if not isinstance(obj, dict):
        return ["example must be an object"]
    for key in REQUIRED_TOP:
        if key not in obj:
            errors.append(f"missing field: {key}")
    if "goal" in obj and not isinstance(obj["goal"], str):
        errors.append("goal must be a string")
    if "context" in obj and not isinstance(obj["context"], str):
        errors.append("context must be a string")
    if "tools" in obj and not isinstance(obj["tools"], list):
        errors.append("tools must be an array")
    resp = obj.get("response")
    if resp is not None:
        if not isinstance(resp, dict):
            errors.append("response must be an object")
        else:
            for key in REQUIRED_RESPONSE:
                if key not in resp:
                    errors.append(f"response missing field: {key}")
            if "plan" in resp and not isinstance(resp["plan"], list):
                errors.append("response.plan must be an array")
            if "verification" in resp and not isinstance(resp["verification"], list):
                errors.append("response.verification must be an array")
    return errors


def validate_jsonl(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    total = 0
    valid = 0
    problems: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                problems.append({"line": i, "errors": [f"json: {e}"]})
                continue
            errs = validate_instruction_example(obj)
            if errs:
                problems.append({"line": i, "errors": errs})
            else:
                valid += 1
    return {"total": total, "valid": valid, "invalid": total - valid, "problems": problems[:50]}
