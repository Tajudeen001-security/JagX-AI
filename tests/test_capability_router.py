from __future__ import annotations

from inference.capability_router import list_public_capabilities, resolve_route


def test_public_ids_do_not_look_like_upstream():
    for c in list_public_capabilities():
        assert c["id"].startswith("jagx-")
        assert "/" not in c["id"]


def test_code_heuristic():
    r = resolve_route(prompt="implement a python function to sort")
    assert r.public_id == "jagx-code"
    assert "/" in r.upstream_model  # internal only


def test_vision_forced():
    r = resolve_route(has_image=True)
    assert r.supports_vision
    assert r.public_id.startswith("jagx-vision")
