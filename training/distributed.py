from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from model import JagXTransformer, ModelConfig
from tokenizer import JagXTokenizer
from training.entrypoint import load_examples
from training.pretraining import PretrainingConfig, packed_batches, prepare_examples


def setup() -> tuple[int, int, torch.device]:
    if not torch.cuda.is_available():
        raise RuntimeError("distributed training requires CUDA")
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    return rank, world, torch.device("cuda", local_rank)


def main() -> None:
    parser = argparse.ArgumentParser(description="JagX multi-GPU DDP pretraining")
    parser.add_argument("--data", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", default="checkpoints/ddp")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rank, world, device = setup()
    torch.manual_seed(args.seed + rank)

    model_cfg = ModelConfig.from_dict(json.loads(Path(args.config).read_text(encoding="utf-8")))
    tokenizer = JagXTokenizer.from_pretrained(args.tokenizer)
    if model_cfg.vocab_size != tokenizer.vocab_size:
        raise ValueError("model vocab_size does not match tokenizer vocabulary")
    model = JagXTransformer(model_cfg).to(device)
    ddp = DDP(model, device_ids=[device.index], broadcast_buffers=False)
    optimizer = torch.optim.AdamW(ddp.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    examples = load_examples(args.data)
    examples, _ = prepare_examples([x for x in examples if x.split == "train"], seed=args.seed)
    cfg = PretrainingConfig(
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        max_steps=args.steps,
        grad_accum=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
    ).validate()
    batches = packed_batches(examples, tokenizer, cfg)

    optimizer.zero_grad(set_to_none=True)
    step = 0
    micro = 0
    for global_index, batch in enumerate(batches):
        if global_index % world != rank:
            continue
        moved = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
        _, loss = ddp(**moved)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"rank {rank}: non-finite loss")
        (loss / args.grad_accum).backward()
        micro += 1
        if micro % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(ddp.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if rank == 0 and step % 10 == 0:
                print(json.dumps({"step": step, "loss": float(loss.detach().item()), "world_size": world}), flush=True)
            if step >= args.steps:
                break

    if micro % args.grad_accum != 0 and step < args.steps:
        torch.nn.utils.clip_grad_norm_(ddp.parameters(), 1.0)
        optimizer.step()
        step += 1

    dist.barrier()
    if rank == 0:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        torch.save({"model": ddp.module.state_dict(), "config": model_cfg.to_dict(), "step": step}, out / "model.pt")
        (out / "distributed_metadata.json").write_text(json.dumps({"world_size": world, "steps": step}, indent=2), encoding="utf-8")
        print(f"Saved distributed checkpoint to {out / 'model.pt'}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
