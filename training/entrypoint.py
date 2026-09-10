from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import torch

from model import JagXTransformer, ModelConfig
from tokenizer import JagXTokenizer

from .data_contract import TrainingExample
from .pretraining import PretrainingConfig, build_optimizer, build_scheduler, packed_batches, prepare_examples
from .seed import set_seed
from .trainer import CausalLMTrainer, TrainerConfig


def load_examples(path: str | Path) -> list[TrainingExample]:
    """Load validated text examples from JSONL."""
    source_path = Path(path)
    if not source_path.is_file():
        raise FileNotFoundError(f"training data not found: {source_path}")
    examples: list[TrainingExample] = []
    with source_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"line {line_number} must contain a JSON object")
            examples.append(TrainingExample(text=str(row.get("text", "")), source=str(row.get("source", source_path.name)), split=str(row.get("split", "train")), quality=float(row.get("quality", 1.0)), license=str(row.get("license", "unknown"))).validate())
    if not examples:
        raise ValueError("training data is empty")
    return examples


def build_model(model_config: ModelConfig, tokenizer: JagXTokenizer) -> JagXTransformer:
    model_config.validate()
    if model_config.vocab_size != tokenizer.vocab_size:
        raise ValueError(f"model vocab_size ({model_config.vocab_size}) does not match tokenizer ({tokenizer.vocab_size})")
    return JagXTransformer(model_config)


def evaluate_loss(model: torch.nn.Module, batches: Iterable[dict], device: torch.device, max_batches: int = 32) -> float:
    if max_batches < 1:
        raise ValueError("max_batches must be positive")
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for batch in batches:
            moved = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
            output = model(**moved)
            loss = output[1] if isinstance(output, (tuple, list)) else output["loss"] if isinstance(output, dict) else output.loss
            if loss.numel() != 1:
                loss = loss.mean()
            if not torch.isfinite(loss).all():
                raise FloatingPointError("non-finite validation loss")
            total += float(loss.detach().item())
            count += 1
            if count >= max_batches:
                break
    model.train()
    if not count:
        raise ValueError("validation set produced no batches")
    return total / count


def _resume_lr_ratio(resume_from: str | Path | None, learning_rate: float, max_steps: int) -> tuple[int, float] | None:
    if not resume_from:
        return None
    path = Path(resume_from)
    if not path.is_file():
        return None
    state = torch.load(path, map_location="cpu", weights_only=True)
    step = int(state.get("step", 0))
    optimizer = state.get("optimizer")
    groups = optimizer.get("param_groups", []) if isinstance(optimizer, dict) else []
    if step >= max_steps or not groups:
        return None
    current_lr = float(groups[0].get("lr", learning_rate))
    return step, max(1e-8, current_lr / learning_rate)


def run_training(data_path: str | Path, tokenizer_path: str | Path, model_config: ModelConfig, pretraining_config: PretrainingConfig, output_dir: str | Path = "checkpoints", resume_from: str | Path | None = None, validation_data_path: str | Path | None = None, device: str | None = None) -> dict:
    """Run native training. AMP is opt-in; FP32 is the stable Kaggle default."""
    cfg = pretraining_config.validate()
    set_seed(cfg.seed)
    tokenizer = JagXTokenizer.from_pretrained(tokenizer_path)
    target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_config, tokenizer)

    gpu_count = torch.cuda.device_count() if target_device == "cuda" else 0
    use_multi_gpu = gpu_count > 1
    if use_multi_gpu:
        print(f"Using DataParallel across {gpu_count} CUDA GPUs")
        model = torch.nn.DataParallel(model, device_ids=list(range(gpu_count)))

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    resume_schedule = _resume_lr_ratio(resume_from, cfg.learning_rate, cfg.max_steps)
    if resume_schedule is not None:
        resume_step, resume_ratio = resume_schedule
        scheduler.lr_lambdas[0] = lambda _step, ratio=resume_ratio: ratio
        print(f"Extending from step {resume_step}: preserving checkpoint learning rate ({resume_ratio:.8g} x base LR)")

    use_amp = target_device == "cuda" and os.environ.get("JAGX_USE_AMP", "0") == "1"
    print("AMP enabled by JAGX_USE_AMP=1" if use_amp else "AMP disabled: using stable FP32 training")

    trainer = CausalLMTrainer(model, optimizer, scheduler=scheduler, config=TrainerConfig(max_steps=cfg.max_steps, grad_accum=cfg.grad_accum, output_dir=str(output_dir), device=device, use_amp=use_amp), resume_from=str(resume_from) if resume_from else None)

    examples = load_examples(data_path)
    train_examples, validation_examples = [], []
    for example in examples:
        if example.split == "validation":
            validation_examples.append(example)
        elif example.split == "train":
            train_examples.append(example)
    train_examples, _ = prepare_examples(train_examples, seed=cfg.seed)
    if not train_examples:
        raise ValueError("no training examples remain after filtering")

    metrics = trainer.train(packed_batches(train_examples, tokenizer, cfg))
    result = {"train": metrics, "step": trainer.step, "model_config": model_config.to_dict(), "pretraining_config": asdict(cfg), "learning_rate": optimizer.param_groups[0]["lr"], "gpu_count": gpu_count, "multi_gpu": use_multi_gpu, "amp": use_amp}

    if validation_data_path is not None:
        validation_examples = load_examples(validation_data_path)
    if validation_examples:
        validation_examples, _ = prepare_examples(validation_examples, seed=cfg.seed)
        if validation_examples:
            result["validation_loss"] = evaluate_loss(model, packed_batches(validation_examples, tokenizer, cfg), trainer.device)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the native JagX causal language model")
    parser.add_argument("--data", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--config")
    parser.add_argument("--validation-data")
    parser.add_argument("--resume")
    parser.add_argument("--out-dir", default="checkpoints")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vocab-size", type=int, default=32768)
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--hidden-size", type=int, default=384)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default=None)
    args = parser.parse_args()
    model_config = ModelConfig.from_dict(json.loads(Path(args.config).read_text(encoding="utf-8"))) if args.config else ModelConfig(vocab_size=args.vocab_size, max_seq_len=args.context_length, d_model=args.hidden_size, n_layers=args.layers, n_heads=args.heads)
    pretraining_config = PretrainingConfig(seq_len=args.seq_len, batch_size=args.batch_size, max_steps=args.steps, grad_accum=args.grad_accum, learning_rate=args.lr, weight_decay=args.weight_decay, warmup_steps=args.warmup_steps, min_lr_ratio=args.min_lr_ratio, seed=args.seed)
    result = run_training(args.data, args.tokenizer, model_config, pretraining_config, output_dir=args.out_dir, resume_from=args.resume, validation_data_path=args.validation_data, device=args.device)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
