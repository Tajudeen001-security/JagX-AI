"""Unbound generate must not fake success with empty text."""

from __future__ import annotations

from api.server import _unbound_generate


def test_unbound_generate_is_explicit_error():
    out = _unbound_generate({"prompt": "hi"})
    assert out.get("error") == "model_not_bound"
    assert "text" not in out or out.get("text") in (None, "")
