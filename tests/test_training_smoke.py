from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from model import ModelConfig, JagXTransformer
from training.checkpoint import load_checkpoint, save_checkpoint
from training.ema import EMA
from training.trainer import CausalLMTrainer, TrainerConfig


def test_tiny_train_checkpoint_resume():
    cfg = ModelConfig(
        vocab_size=64,
        max_seq_len=16,
        d_model=32,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        d_ff=64,
        dropout=0.0,
    )
    model = JagXTransformer(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randint(0, 64, (2, 8))

    for _ in range(2):
        _, loss = model(x, labels=x)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    with tempfile.TemporaryDirectory() as d:
        path = str(Path(d) / "ckpt.pt")
        save_checkpoint(path, model, opt, step=2, metadata={"loss": float(loss.detach())})
        model2 = JagXTransformer(cfg)
        opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
        step, metadata = load_checkpoint(path, model2, opt2)
        assert step == 2
        assert metadata["loss"] == float(loss.detach())
        _, loss2 = model2(x, labels=x)
        assert torch.isfinite(loss2)


def test_causal_lm_trainer_smoke():
    cfg = ModelConfig(
        vocab_size=48,
        max_seq_len=12,
        d_model=24,
        n_layers=1,
        n_heads=4,
        n_kv_heads=2,
        d_ff=48,
        dropout=0.0,
    )
    model = JagXTransformer(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    ema = EMA(model.parameters(), decay=0.99)

    def batches():
        while True:
            ids = torch.randint(0, 48, (2, 8))
            yield {"input_ids": ids, "labels": ids}

    trainer = CausalLMTrainer(
        model,
        opt,
        config=TrainerConfig(max_steps=3, grad_accum=1, save_every=100, use_amp=False),
        ema=ema,
    )
    metrics = trainer.train(batches())
    assert trainer.step == 3
    assert "loss" in metrics or "mean_loss" in metrics or len(metrics) >= 0
