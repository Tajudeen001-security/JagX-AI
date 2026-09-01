from dataclasses import dataclass, asdict
import json
from pathlib import Path


@dataclass
class TrainConfig:
    seed: int = 42
    batch_size: int = 2
    grad_accum: int = 8
    steps: int = 1000
    lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 100
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_every: int = 100
    save_every: int = 100
    out_dir: str = "checkpoints"
    amp: bool = True
    # Scaling controls: increase these progressively as hardware permits.
    vocab_size: int = 65536
    context_length: int = 4096
    hidden_size: int = 768
    layers: int = 12
    heads: int = 12
    gradient_checkpointing: bool = True
    bf16: bool = True

    def validate(self):
        if min(self.batch_size, self.grad_accum, self.steps, self.eval_every, self.save_every) <= 0:
            raise ValueError("training counts must be positive")
        if min(self.lr, self.min_lr, self.weight_decay, self.grad_clip) <= 0:
            raise ValueError("training hyperparameters must be positive")
        if self.min_lr > self.lr:
            raise ValueError("min_lr cannot exceed lr")
        if min(self.vocab_size, self.context_length, self.hidden_size, self.layers, self.heads) <= 0:
            raise ValueError("model dimensions must be positive")
        if self.hidden_size % self.heads:
            raise ValueError("hidden_size must be divisible by heads")
        if self.bf16 and not self.amp:
            raise ValueError("bf16 requires amp")
        return self

    def save(self, path):
        self.validate()
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(self), indent=2))
