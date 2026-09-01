from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import torch

from .amp import autocast_context
from .checkpoint import save_checkpoint
from .ema import EMA
from .metrics import RunningMetrics


@dataclass(frozen=True)
class TrainerConfig:
    max_steps: int = 1000
    grad_accum: int = 8
    grad_clip: float = 1.0
    log_every: int = 10
    save_every: int = 100
    output_dir: str = "checkpoints"
    use_amp: bool = True

    def validate(self) -> "TrainerConfig":
        if self.max_steps <= 0 or self.grad_accum <= 0:
            raise ValueError("max_steps and grad_accum must be positive")
        if self.grad_clip <= 0 or self.log_every <= 0 or self.save_every <= 0:
            raise ValueError("grad_clip, log_every and save_every must be positive")
        return self


class CausalLMTrainer:
    """Model-agnostic PyTorch trainer for causal language-model objectives.

    Batch contract: either a dict passed as kwargs, or a tensor/tuple.
    The model may return a scalar loss, a (logits, loss) tuple, or an object/dict
    containing ``loss``. This keeps JagX independent of any external provider.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any = None,
        config: TrainerConfig | None = None,
        ema: EMA | None = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = (config or TrainerConfig()).validate()
        self.ema = ema
        self.metrics = RunningMetrics()
        self.step = 0

    @staticmethod
    def _loss(output: Any) -> torch.Tensor:
        if torch.is_tensor(output):
            return output
        if isinstance(output, (tuple, list)) and len(output) >= 2:
            # (logits, loss) or (logits, loss, cache)
            candidate = output[1]
            if torch.is_tensor(candidate):
                return candidate
        if isinstance(output, dict) and "loss" in output:
            return output["loss"]
        if hasattr(output, "loss"):
            return output.loss
        raise TypeError("model output must be a loss tensor, (logits, loss), or contain a 'loss' field")

    @staticmethod
    def _tokens(batch: Any) -> int:
        if isinstance(batch, dict):
            for key in ("labels", "input_ids"):
                value = batch.get(key)
                if torch.is_tensor(value):
                    return int(value.numel())
        if torch.is_tensor(batch):
            return int(batch.numel())
        return 0

    def train(self, batches: Iterable[Any]) -> dict:
        self.model.train()
        iterator = iter(batches)
        while self.step < self.config.max_steps:
            self.optimizer.zero_grad(set_to_none=True)
            accumulated = 0.0
            tokens = 0
            for _ in range(self.config.grad_accum):
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterator = iter(batches)
                    batch = next(iterator)
                with autocast_context(enabled=self.config.use_amp):
                    if isinstance(batch, dict):
                        output = self.model(**batch)
                    else:
                        output = self.model(batch)
                    loss = self._loss(output)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss at step {self.step + 1}")
                (loss / self.config.grad_accum).backward()
                accumulated += float(loss.detach().item())
                tokens += self._tokens(batch)

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            if self.ema is not None:
                self.ema.update(self.model)
            self.step += 1
            mean_loss = accumulated / self.config.grad_accum
            self.metrics.update(mean_loss, tokens)

            if self.step % self.config.save_every == 0:
                save_checkpoint(
                    f"{self.config.output_dir}/step-{self.step}.pt",
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    self.step,
                    metadata=self.metrics.snapshot(),
                )

        return self.metrics.snapshot()
