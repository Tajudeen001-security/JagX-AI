from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from model import JagXTransformer, ModelConfig
from training.checkpoint import CHECKPOINT_FORMAT_VERSION, load_checkpoint, save_checkpoint
from training.ema import EMA
from training.trainer import CausalLMTrainer, TrainerConfig


def _model():
    return JagXTransformer(
        ModelConfig(
            vocab_size=32,
            max_seq_len=8,
            d_model=16,
            n_layers=1,
            n_heads=4,
            n_kv_heads=2,
            d_ff=32,
            dropout=0.0,
        )
    )


def test_checkpoint_restores_optimizer_scheduler_step_metadata_and_ema():
    model = _model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    ema = EMA(model.parameters(), decay=0.9)
    x = torch.randint(0, 32, (2, 8))
    _, loss = model(x, labels=x)
    loss.backward()
    optimizer.step()
    scheduler.step()
    ema.update(model)

    with tempfile.TemporaryDirectory() as directory:
        path = str(Path(directory) / "resume.pt")
        save_checkpoint(path, model, optimizer, scheduler, step=7, metadata={"mean_loss": float(loss)}, ema=ema)

        restored_model = _model()
        restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=1e-3)
        restored_scheduler = torch.optim.lr_scheduler.StepLR(restored_optimizer, step_size=1, gamma=0.5)
        restored_ema = EMA(restored_model.parameters(), decay=0.5)
        step, metadata = load_checkpoint(
            path,
            restored_model,
            restored_optimizer,
            restored_scheduler,
            ema=restored_ema,
        )

        assert step == 7
        assert metadata["mean_loss"] == float(loss)
        assert restored_scheduler.last_epoch == scheduler.last_epoch
        assert restored_optimizer.state_dict()["state"]
        for original, restored in zip(model.parameters(), restored_model.parameters()):
            assert torch.equal(original, restored)
        for original, restored in zip(ema.shadow.values(), restored_ema.shadow.values()):
            assert torch.equal(original, restored)
        assert not list(Path(directory).glob("*.tmp"))


def test_trainer_resume_continues_from_saved_step():
    with tempfile.TemporaryDirectory() as directory:
        first = _model()
        first_optimizer = torch.optim.AdamW(first.parameters(), lr=1e-3)
        first_trainer = CausalLMTrainer(
            first,
            first_optimizer,
            config=TrainerConfig(max_steps=2, grad_accum=1, save_every=100, use_amp=False, device="cpu"),
        )
        batches = [{"input_ids": torch.ones(2, 8, dtype=torch.long), "labels": torch.ones(2, 8, dtype=torch.long)}]
        first_trainer.train(batches)
        checkpoint = str(Path(directory) / "step-2.pt")
        first_trainer.save(checkpoint)

        resumed = _model()
        resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=1e-3)
        resumed_trainer = CausalLMTrainer(
            resumed,
            resumed_optimizer,
            config=TrainerConfig(max_steps=4, grad_accum=1, save_every=100, use_amp=False, device="cpu"),
            resume_from=checkpoint,
        )
        assert resumed_trainer.step == 2
        resumed_trainer.train(batches)
        assert resumed_trainer.step == 4
        assert resumed_trainer.metrics.steps == 4


def test_checkpoint_rejects_future_format():
    model = _model()
    with tempfile.TemporaryDirectory() as directory:
        path = str(Path(directory) / "future.pt")
        torch.save({"format_version": CHECKPOINT_FORMAT_VERSION + 1, "step": 1, "model": model.state_dict()}, path)
        with pytest.raises(ValueError, match="unsupported JagX checkpoint format"):
            load_checkpoint(path, model)


def test_checkpoint_rejects_negative_step():
    model = _model()
    with tempfile.TemporaryDirectory() as directory:
        path = str(Path(directory) / "invalid.pt")
        torch.save({"format_version": CHECKPOINT_FORMAT_VERSION, "step": -1, "model": model.state_dict()}, path)
        with pytest.raises(ValueError, match="step must be non-negative"):
            load_checkpoint(path, model)
