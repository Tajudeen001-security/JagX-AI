from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator

import torch

from tokenizer import JagXTokenizer

from .data_contract import TrainingExample
from .data_pipeline import CorpusPipeline
from .trainer import CausalLMTrainer, TrainerConfig


@dataclass(frozen=True)
class PretrainingConfig:
    seq_len: int = 512
    batch_size: int = 4
    seed: int = 42
    max_steps: int = 1000
    grad_accum: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 0
    min_lr_ratio: float = 0.1
    drop_remainder: bool = True

    def validate(self) -> "PretrainingConfig":
        if self.seq_len < 2 or self.batch_size < 1:
            raise ValueError("seq_len must be >= 2 and batch_size must be positive")
        if self.max_steps < 1 or self.grad_accum < 1:
            raise ValueError("max_steps and grad_accum must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative")
        if self.warmup_steps < 0 or self.warmup_steps > self.max_steps:
            raise ValueError("warmup_steps must be between 0 and max_steps")
        if not 0.0 < self.min_lr_ratio <= 1.0:
            raise ValueError("min_lr_ratio must be in (0, 1]")
        return self


def token_stream(examples: Iterable[TrainingExample], tokenizer: JagXTokenizer) -> Iterator[int]:
    """Yield tokenized documents with an EOS boundary between documents."""
    for example in examples:
        ids = tokenizer.encode(example.text, add_special_tokens=False)
        if ids:
            yield from ids
        yield tokenizer.eos_token_id


def _iter_packed_batches(
    examples: Iterable[TrainingExample], tokenizer: JagXTokenizer, config: PretrainingConfig
) -> Iterator[dict[str, torch.Tensor]]:
    cfg = config.validate()
    buffer: list[int] = []
    batch: list[list[int]] = []
    pad = tokenizer.pad_token_id

    for token_id in token_stream(examples, tokenizer):
        buffer.append(token_id)
        while len(buffer) >= cfg.seq_len:
            batch.append(buffer[: cfg.seq_len])
            del buffer[: cfg.seq_len]
            if len(batch) == cfg.batch_size:
                values = torch.tensor(batch, dtype=torch.long)
                yield {"input_ids": values, "labels": values.clone()}
                batch.clear()

    if not cfg.drop_remainder and buffer:
        batch.append(buffer + [pad] * (cfg.seq_len - len(buffer)))

    if not cfg.drop_remainder and batch:
        while len(batch) < cfg.batch_size:
            batch.append([pad] * cfg.seq_len)
        values = torch.tensor(batch, dtype=torch.long)
        labels = values.clone()
        labels[values == pad] = -100
        yield {"input_ids": values, "labels": labels}


class PackedBatchIterable:
    """Re-iterable packed-batch stream for multi-epoch training.

    The trainer restarts its input iterable when it reaches the end. A raw
    generator cannot be restarted, which caused long runs to fail with
    ``training batches are empty`` after the first pass. This wrapper rebuilds
    the lightweight iterator without materializing all batches in RAM.
    """

    def __init__(self, examples: Iterable[TrainingExample], tokenizer: JagXTokenizer, config: PretrainingConfig):
        self.examples = examples
        self.tokenizer = tokenizer
        self.config = config.validate()

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        return _iter_packed_batches(self.examples, self.tokenizer, self.config)


def packed_batches(
    examples: Iterable[TrainingExample], tokenizer: JagXTokenizer, config: PretrainingConfig
) -> PackedBatchIterable:
    """Return a re-iterable stream of fixed-length next-token batches."""
    return PackedBatchIterable(examples, tokenizer, config)


def prepare_examples(
    examples: list[TrainingExample], *, seed: int = 42, rank: int = 0, world_size: int = 1
) -> tuple[list[TrainingExample], object]:
    """Apply quality filtering, deduplication and deterministic sharding."""
    return CorpusPipeline().process(examples, seed=seed, rank=rank, world_size=world_size)


def build_optimizer(model: torch.nn.Module, config: PretrainingConfig) -> torch.optim.AdamW:
    """Create AdamW with explicit beta/epsilon values used by the training recipe."""
    cfg = config.validate()
    return torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.95),
        eps=1e-8,
    )


def build_scheduler(optimizer: torch.optim.Optimizer, config: PretrainingConfig):
    """Build a warmup + cosine decay scheduler with a non-zero LR floor."""
    cfg = config.validate()

    def schedule(step: int) -> float:
        if step < cfg.warmup_steps and cfg.warmup_steps:
            return max(1e-8, float(step + 1) / cfg.warmup_steps)
        if step >= cfg.max_steps:
            return cfg.min_lr_ratio
        progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
        cosine = 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi)).item())
        return cfg.min_lr_ratio + (1.0 - cfg.min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)


def train_causal_lm(
    model: torch.nn.Module,
    tokenizer: JagXTokenizer,
    examples: list[TrainingExample],
    config: PretrainingConfig | None = None,
    trainer_config: TrainerConfig | None = None,
) -> dict:
    """Run native JagX causal-LM pretraining from validated text examples."""
    cfg = (config or PretrainingConfig()).validate()
    processed, _ = prepare_examples(examples, seed=cfg.seed)
    if not processed:
        raise ValueError("no training examples remain after corpus filtering")
    batches = packed_batches(processed, tokenizer, cfg)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    tc = trainer_config or TrainerConfig(max_steps=cfg.max_steps, grad_accum=cfg.grad_accum)
    trainer = CausalLMTrainer(model, optimizer, scheduler=scheduler, config=tc)
    return trainer.train(batches)
