from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from model import ModelConfig, JagXTransformer
from training.seed import set_seed


def load_config(args) -> ModelConfig:
    if args.config:
        data = json.loads(Path(args.config).read_text(encoding="utf-8"))
        return ModelConfig.from_dict(data)
    return ModelConfig(
        vocab_size=args.vocab_size,
        max_seq_len=args.seq_len,
        d_model=args.d_model,
        n_layers=args.layers,
        n_heads=args.heads,
        n_kv_heads=args.kv_heads,
        d_ff=args.ff,
        use_swiglu=not args.no_swiglu,
        use_rms_norm=not args.no_rms,
    )


def train(args) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    cfg = load_config(args)
    model = JagXTransformer(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1, betas=(0.9, 0.95))

    token_path = Path(args.tokens)
    if not token_path.exists():
        raise FileNotFoundError(f"Token file not found: {token_path}")
    tokens = torch.tensor(list(map(int, token_path.read_text().split())), dtype=torch.long)
    if len(tokens) < args.seq_len + 1:
        raise ValueError("Not enough tokens for one sequence")

    model.train()
    for step in range(args.steps):
        span = args.batch_size * args.seq_len
        start = (step * span) % max(1, len(tokens) - span - 1)
        x = tokens[start : start + span].view(args.batch_size, args.seq_len).to(device)
        y = tokens[start + 1 : start + 1 + span].view(args.batch_size, args.seq_len).to(device)
        _, loss = model(x, labels=y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % args.log_every == 0:
            print(f"step={step} loss={loss.item():.4f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": cfg.to_dict()}, out)
    print(f"Saved checkpoint to {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Minimal JagX causal LM training loop")
    p.add_argument("--tokens", required=True, help="Whitespace-separated token id file")
    p.add_argument("--out", default="checkpoints/jagx.pt")
    p.add_argument("--config", default=None, help="JSON model config (overrides CLI size flags)")
    p.add_argument("--vocab-size", type=int, default=32768)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--layers", type=int, default=8)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--kv-heads", type=int, default=None)
    p.add_argument("--ff", type=int, default=None)
    p.add_argument("--no-swiglu", action="store_true")
    p.add_argument("--no-rms", action="store_true")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")
    train(p.parse_args())
