from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import math
import random
import torch
from .checkpoint import save_checkpoint, load_checkpoint
from .schedule import cosine_lr

@dataclass
class TrainConfig:
    steps: int = 1000
    grad_accum: int = 1
    lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    log_every: int = 10
    checkpoint_every: int = 500
    checkpoint_path: str = "checkpoints/latest.pt"
    seed: int = 42

class Trainer:
    def __init__(self, model, optimizer, config: TrainConfig, scheduler=None):
        self.model=model
        self.optimizer=optimizer
        self.config=config
        self.scheduler=scheduler
        self.device=next(model.parameters()).device
        self.use_amp=self.device.type == "cuda"
        self.scaler=torch.amp.GradScaler("cuda", enabled=self.use_amp)

    def fit(self, batches: Iterable[tuple[torch.Tensor, torch.Tensor]], start_step: int = 0):
        torch.manual_seed(self.config.seed); random.seed(self.config.seed)
        self.model.train(); iterator=iter(batches)
        for step in range(start_step, self.config.steps):
            self.optimizer.zero_grad(set_to_none=True)
            total_loss=0.0
            for _ in range(self.config.grad_accum):
                x,y=next(iterator); x=x.to(self.device); y=y.to(self.device)
                with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp):
                    _,loss=self.model(x,y); loss=loss/self.config.grad_accum
                self.scaler.scale(loss).backward(); total_loss += float(loss.detach())
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.scaler.step(self.optimizer); self.scaler.update()
            lr=cosine_lr(step,self.config.steps,self.config.warmup_steps,self.config.lr,self.config.min_lr)
            for group in self.optimizer.param_groups: group["lr"]=lr
            if step % self.config.log_every == 0:
                print(f"step={step} loss={total_loss:.4f} lr={lr:.6g}")
            if self.config.checkpoint_every and (step+1)%self.config.checkpoint_every==0:
                save_checkpoint(self.config.checkpoint_path,self.model,self.optimizer,self.scheduler,step+1,{"config":self.config.__dict__})
        return step + 1
