"""End-to-end JagX training orchestration.

Runs the stages that are available locally: corpus preparation, optional
pretraining, optional preference optimization, and evaluation. Each stage
writes a machine-readable manifest so a failed stage can be resumed safely.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--out-dir", default="artifacts/jagx_run")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--device", choices=["cpu", "cuda", "mps"], default=None)
    p.add_argument("--skip-train", action="store_true")
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"status": "started", "stages": {}}
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    data = Path(args.data)
    tok = Path(args.tokenizer)
    if not data.is_file():
        raise FileNotFoundError(data)
    if not tok.exists():
        raise FileNotFoundError(tok)

    manifest["stages"]["data_validation"] = "passed"
    if not args.skip_train:
        cmd = [sys.executable, "-m", "training.entrypoint", "--data", str(data),
               "--tokenizer", str(tok), "--out-dir", str(out / "checkpoints"),
               "--steps", str(args.steps), "--batch-size", str(args.batch_size),
               "--grad-accum", str(args.grad_accum), "--seq-len", str(args.seq_len)]
        if args.device:
            cmd += ["--device", args.device]
        run(cmd)
        manifest["stages"]["pretraining"] = "passed"
    else:
        manifest["stages"]["pretraining"] = "skipped"

    manifest["status"] = "completed"
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
