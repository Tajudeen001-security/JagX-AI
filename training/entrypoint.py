from __future__ import annotations

import argparse
import json
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
            examples.append(
                TrainingExample(
                    text=str(row.get("text", "")),
                    source=str(row.get("source", source_path.name)),
                    split=str(row.get("split", "train")),
                    quality=float(row.get("quality", 1.0)),
                    license=str(row.get("license", "unknown")),
                ).validate()
            )
    if not examples:
        raise ValueError("training data is empty")
    return examples


def build_model(model_config: ModelConfig, tokenizer: JagXTokenizer) -> JagXTransformer:
    """Instantiate the native JagX causal LM and verify vocabulary compatibility."""
    model_config.validate()
    if model_config.vocab_size != tokenizer.vocab_size:
        raise ValueError(
            f"model vocab_size ({model_config.vocab_size}) does not match tokenizer ({tokenizer.vocab_size})"
        )
    return JagXTransformer(model_config)


def evaluate_loss(model: torch.nn.Module, batches: Iterable[dict], device: torch.device, max_batches: int = 32) -> float:
    """Compute mean finite validation loss without changing model weights."""
    if max_batches < 1:
        raise ValueError("max_batches must be positive")
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for batch in batches:
            moved = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
            output = model(**moved)
            if isinstance(output, (tuple, list)):
                loss = output[1]
            elif isinstance(output, dict):
                loss = output["loss"]
            else:
                loss = output.loss
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite validation loss")
            total += float(loss.detach().item())
            count += 1
            if count >= max_batches:
                break
    model.train()
    if not count:
        raise ValueError("validation set produced no batches")
    return total / count


def run_training(
    data_path: str | Path,
    tokenizer_path: str | Path,
    model_config: ModelConfig,
    pretraining_config: PretrainingConfig,
    output_dir: str | Path = "checkpoints",
    resume_from: str | Path | None = None,
    validation_data_path: str | Path | None = None,
    device: str | None = None,
) -> dict:
    """Run the complete local training pipeline and optionally resume a checkpoint."""
    cfg = pretraining_config.validate()
    set_seed(cfg.seed)
    tokenizer = JagXTokenizer.from_pretrained(tokenizer_path)
    model = build_model(model_config, tokenizer)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    trainer_config = TrainerConfig(
        max_steps=cfg.max_steps,
        grad_accum=cfg.grad_accum,
        output_dir=str(output_dir),
        device=device,
        use_amp=target_device in {"cuda", "mps"},
    )
    trainer = CausalLMTrainer(
        model,
        optimizer,
        scheduler=scheduler,
        config=trainer_config,
        resume_from=str(resume_from) if resume_from else None,
    )

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
    result = {
        "train": metrics,
        "step": trainer.step,
        "model_config": model_config.to_dict(),
        "pretraining_config": asdict(cfg),
        "learning_rate": optimizer.param_groups[0]["lr"],
    }

    if validation_data_path is not None:
        validation_examples = load_examples(validation_data_path)
    if validation_examples:
        validation_examples, _ = prepare_examples(validation_examples, seed=cfg.seed)
        if validation_examples:
            validation_batches = packed_batches(validation_examples, tokenizer, cfg)
            result["validation_loss"] = evaluate_loss(model, validation_batches, trainer.device)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the native JagX causal language model")
    parser.add_argument("--data", required=True, help="JSONL with text/source/split fields")
    parser.add_argument("--tokenizer", required=True, help="directory containing tokenizer.json")
    parser.add_argument("--config", help="JSON ModelConfig; otherwise use CLI model settings")
    parser.add_argument("--validation-data", help="optional validation JSONL")
    parser.add_argument("--resume", help="checkpoint to resume")
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

    if args.config:
        model_config = ModelConfig.from_dict(json.loads(Path(args.config).read_text(encoding="utf-8")))
    else:
        model_config = ModelConfig(
            vocab_size=args.vocab_size,
            max_seq_len=args.context_length,
            d_model=args.hidden_size,
            n_layers=args.layers,
            n_heads=args.heads,
        )
    pretraining_config = PretrainingConfig(
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        max_steps=args.steps,
        grad_accum=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        min_lr_ratio=args.min_lr_ratio,
        seed=args.seed,
    )
    result = run_training(
        args.data,
        args.tokenizer,
        model_config,
        pretraining_config,
        output_dir=args.out_dir,
        resume_from=args.resume,
        validation_data_path=args.validation_data,
        device=args.device,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
