from __future__ import annotations

import os

from inference.nvidia_client import NvidiaClient, public_models_list


def test_public_models_list_shape():
    out = public_models_list()
    assert out["object"] == "list"
    assert all("id" in m for m in out["data"])
    assert all("/" not in m["id"] for m in out["data"])


def test_client_unavailable_without_keys(monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("NVIDIA_API_KEYS", raising=False)
    c = NvidiaClient()
    assert c.available is False
