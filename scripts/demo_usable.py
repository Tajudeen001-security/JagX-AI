#!/usr/bin/env python3
"""One-command demo: proves the stack is usable without pretrained weights."""

from __future__ import annotations

import json
import sys


def main() -> int:
    print("=== JagX usable demo (no pretrained weights required) ===\n")

    from runtime.orchestrator import RequestStatus, build_default_orchestrator

    orch = build_default_orchestrator()

    # 1) Health
    h = orch.execute({"kind": "health"})
    assert h.status == RequestStatus.SUCCEEDED, h.error
    print("[1] Health OK")
    print("    capabilities:", [c["name"] for c in h.data.get("capabilities", [])])

    # 2) Memory
    from memory.store import MemoryStore

    store = MemoryStore()
    a = orch.execute(
        {
            "kind": "memory",
            "memory_action": "add",
            "content": "JagX agent uses a TaskDAG planner with retries and receipts.",
            "_store": store,
        }
    )
    assert a.status == RequestStatus.SUCCEEDED, a.error
    r = orch.execute(
        {
            "kind": "memory",
            "memory_action": "retrieve",
            "query": "TaskDAG planner",
            "_store": store,
        }
    )
    assert r.status == RequestStatus.SUCCEEDED, r.error
    hits = r.data.get("hits") or []
    print("[2] Memory OK — hits:", len(hits))

    # 3) Agent DAG
    ag = orch.execute({"goal": "implement and test a small utility function", "timeout_s": 60})
    assert ag.status == RequestStatus.SUCCEEDED, ag.error
    data = ag.data or {}
    print("[3] Agent OK — backend:", data.get("backend") or data.get("status"))
    dag = data.get("dag") or {}
    if dag:
        print("    tasks:", dag.get("total"), "counts:", dag.get("counts"))

    # 4) Model smoke (structure only)
    from evaluation.model_smoke import run_smoke_test

    smoke = run_smoke_test()
    print("[4] Model smoke OK — params:", smoke["parameters"], "loss:", round(smoke["loss"], 4))

    print("\n=== Demo finished successfully ===")
    print("Next: train a tiny checkpoint (docs/FREE_TRAINING.md) for real text quality.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("DEMO FAILED:", exc, file=sys.stderr)
        raise SystemExit(1)
