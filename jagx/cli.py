"""Real CLI for JagX — no demo mode."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def cmd_verify(_args: argparse.Namespace) -> int:
    """Hard system verification: real modules, real agent DAG, real tools."""
    from runtime.orchestrator import RequestStatus, build_default_orchestrator
    from evaluation.model_smoke import run_smoke_test
    from memory.store import MemoryStore

    orch = build_default_orchestrator()
    failures: list[str] = []

    h = orch.execute({"kind": "health"})
    if h.status != RequestStatus.SUCCEEDED:
        failures.append(f"health: {h.error}")
    else:
        print("OK health", json.dumps({"capabilities": [c["name"] for c in (h.data or {}).get("capabilities", [])]}))

    store = MemoryStore()
    a = orch.execute(
        {
            "kind": "memory",
            "memory_action": "add",
            "content": "JagX uses TaskDAG execution with receipts.",
            "_store": store,
        }
    )
    r = orch.execute(
        {
            "kind": "memory",
            "memory_action": "retrieve",
            "query": "TaskDAG",
            "_store": store,
        }
    )
    if a.status != RequestStatus.SUCCEEDED or r.status != RequestStatus.SUCCEEDED:
        failures.append(f"memory: {a.error or r.error}")
    else:
        print("OK memory hits=", len((r.data or {}).get("hits") or []))

    t = orch.execute({"kind": "tool", "tool": "echo", "arguments": {"message": "ok"}})
    if t.status != RequestStatus.SUCCEEDED or not (t.data or {}).get("ok"):
        failures.append(f"tool: {t.error or t.data}")
    else:
        print("OK tool", (t.data or {}).get("data"))

    ag = orch.execute({"goal": "research summary pipeline", "timeout_s": 60})
    if ag.status != RequestStatus.SUCCEEDED:
        failures.append(f"agent: {ag.error}")
    else:
        print("OK agent backend=", (ag.data or {}).get("backend"))

    try:
        smoke = run_smoke_test()
        print("OK model_forward params=", smoke["parameters"], "loss=", round(smoke["loss"], 4))
    except Exception as exc:
        failures.append(f"model: {exc}")

    if failures:
        print("VERIFY FAILED:", file=sys.stderr)
        for f in failures:
            print(" -", f, file=sys.stderr)
        return 1
    print("VERIFY PASSED")
    return 0


def cmd_agent(args: argparse.Namespace) -> int:
    from runtime.orchestrator import RequestStatus, build_default_orchestrator

    orch = build_default_orchestrator()
    result = orch.execute({"goal": args.goal, "workspace": args.workspace, "timeout_s": args.timeout})
    print(json.dumps(result.to_dict(), indent=2, default=str))
    return 0 if result.status == RequestStatus.SUCCEEDED else 1


def cmd_memory(args: argparse.Namespace) -> int:
    from runtime.orchestrator import RequestStatus, build_default_orchestrator
    from memory.store import MemoryStore

    store = MemoryStore(args.path) if args.path else MemoryStore()
    orch = build_default_orchestrator()
    if args.action == "add":
        result = orch.execute(
            {
                "kind": "memory",
                "memory_action": "add",
                "content": args.text,
                "durable": args.durable,
                "_store": store,
            }
        )
    else:
        result = orch.execute(
            {
                "kind": "memory",
                "memory_action": "retrieve",
                "query": args.text,
                "k": args.k,
                "_store": store,
            }
        )
    print(json.dumps(result.to_dict(), indent=2, default=str))
    return 0 if result.status == RequestStatus.SUCCEEDED else 1


def cmd_tool(args: argparse.Namespace) -> int:
    from runtime.orchestrator import RequestStatus, build_default_orchestrator

    orch = build_default_orchestrator()
    arguments = json.loads(args.args) if args.args else {}
    result = orch.execute({"kind": "tool", "tool": args.name, "arguments": arguments})
    print(json.dumps(result.to_dict(), indent=2, default=str))
    return 0 if result.status == RequestStatus.SUCCEEDED else 1


def cmd_generate(args: argparse.Namespace) -> int:
    """Real generation only — requires checkpoint + tokenizer."""
    from inference.loader import generate_text, load_model, load_tokenizer

    ckpt = args.checkpoint or os.environ.get("JAGX_CHECKPOINT")
    tok = args.tokenizer or os.environ.get("JAGX_TOKENIZER")
    if not ckpt or not tok:
        print(
            "ERROR: real generation requires --checkpoint and --tokenizer "
            "(or JAGX_CHECKPOINT / JAGX_TOKENIZER). Train first.",
            file=sys.stderr,
        )
        return 2
    if not Path(ckpt).exists():
        print(f"ERROR: checkpoint not found: {ckpt}", file=sys.stderr)
        return 2
    model, _ = load_model(ckpt, device=args.device)
    tokenizer = load_tokenizer(tok)
    text = generate_text(
        model,
        tokenizer,
        args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        seed=args.seed,
    )
    print(text)
    return 0


def cmd_code(args: argparse.Namespace) -> int:
    """Real sandboxed write+test."""
    from runtime.orchestrator import RequestStatus, build_default_orchestrator

    files = json.loads(Path(args.files_json).read_text(encoding="utf-8"))
    orch = build_default_orchestrator()
    result = orch.execute(
        {
            "kind": "code",
            "workspace": args.workspace,
            "files": files,
            "test_command": args.test_command,
            "timeout_s": args.timeout,
        }
    )
    print(json.dumps(result.to_dict(), indent=2, default=str))
    data = result.data or {}
    return 0 if result.status == RequestStatus.SUCCEEDED and data.get("ok") else 1


def cmd_serve(args: argparse.Namespace) -> int:
    from api.server import create_app, serve
    from api.local_generate import make_generate_fn_from_paths

    ckpt = args.checkpoint or os.environ.get("JAGX_CHECKPOINT")
    tok = args.tokenizer or os.environ.get("JAGX_TOKENIZER")
    generate_fn = None
    if ckpt and tok and Path(ckpt).exists():
        generate_fn = make_generate_fn_from_paths(ckpt, tok, device=args.device)
        print(f"Bound real generator: {ckpt}")
    else:
        print("WARN: no checkpoint bound — /v1/generate will return 503 until you train+bind")

    # serve() uses create_app internally; patch via create_app path
    import api.server as server_mod

    def _serve(host: str, port: int) -> None:
        from http.server import HTTPServer

        handler = create_app(generate_fn=generate_fn)
        httpd = HTTPServer((host, port), handler)
        print(f"JagX API http://{host}:{port}")
        httpd.serve_forever()

    _serve(args.host, args.port)
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    """Delegate to real training entrypoint."""
    from training.entrypoint import main as train_main

    # Build argv for training entrypoint if it uses argparse
    argv = [
        "--data",
        args.data,
        "--tokenizer",
        args.tokenizer,
        "--config",
        args.config,
        "--steps",
        str(args.steps),
        "--out-dir",
        args.out_dir,
    ]
    if args.resume:
        argv.extend(["--resume", args.resume])
    if args.device:
        argv.extend(["--device", args.device])
    sys.argv = ["training.entrypoint", *argv]
    try:
        train_main()
        return 0
    except SystemExit as e:
        return int(e.code or 0)
    except Exception as exc:
        print(f"TRAIN ERROR: {exc}", file=sys.stderr)
        return 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="jagx", description="JagX AI — local agent + training platform")
    sub = p.add_subparsers(dest="command", required=True)

    v = sub.add_parser("verify", help="Verify runtime (agent, memory, tools, model forward)")
    v.set_defaults(func=cmd_verify)

    a = sub.add_parser("agent", help="Run agent goal through real TaskDAG")
    a.add_argument("goal")
    a.add_argument("--workspace")
    a.add_argument("--timeout", type=float, default=120.0)
    a.set_defaults(func=cmd_agent)

    m = sub.add_parser("memory", help="Add or retrieve memory")
    m.add_argument("action", choices=["add", "retrieve"])
    m.add_argument("text")
    m.add_argument("--path")
    m.add_argument("--durable", action="store_true")
    m.add_argument("-k", type=int, default=5)
    m.set_defaults(func=cmd_memory)

    t = sub.add_parser("tool", help="Run a registered tool")
    t.add_argument("name")
    t.add_argument("--args", default="{}", help='JSON object, e.g. {"message":"hi"}')
    t.set_defaults(func=cmd_tool)

    g = sub.add_parser("generate", help="Generate text from a real checkpoint")
    g.add_argument("prompt")
    g.add_argument("--checkpoint")
    g.add_argument("--tokenizer")
    g.add_argument("--max-tokens", type=int, default=64)
    g.add_argument("--temperature", type=float, default=0.8)
    g.add_argument("--seed", type=int)
    g.add_argument("--device")
    g.set_defaults(func=cmd_generate)

    c = sub.add_parser("code", help="Write files and run tests in a sandbox")
    c.add_argument("--workspace", required=True)
    c.add_argument("--files-json", required=True, help="JSON file: {relative_path: content}")
    c.add_argument("--test-command", default="python3 -m pytest -q")
    c.add_argument("--timeout", type=float, default=60.0)
    c.set_defaults(func=cmd_code)

    s = sub.add_parser("serve", help="Start local API")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8080)
    s.add_argument("--checkpoint")
    s.add_argument("--tokenizer")
    s.add_argument("--device")
    s.set_defaults(func=cmd_serve)

    tr = sub.add_parser("train", help="Run training entrypoint")
    tr.add_argument("--data", required=True)
    tr.add_argument("--tokenizer", required=True)
    tr.add_argument("--config", default="configs/tiny.json")
    tr.add_argument("--steps", type=int, default=100)
    tr.add_argument("--out-dir", default="checkpoints/run")
    tr.add_argument("--resume")
    tr.add_argument("--device")
    tr.set_defaults(func=cmd_train)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
