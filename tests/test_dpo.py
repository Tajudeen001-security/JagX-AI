import torch
from model import ModelConfig, JagXTransformer
from training.dpo import DPOConfig, DPOTrainerStep, dpo_loss


def test_dpo_loss_shapes():
    chosen = torch.zeros(4)
    rejected = torch.ones(4) * -1
    loss, metrics = dpo_loss(chosen, rejected, None, None, DPOConfig(beta=0.1, reference_free=True))
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert "reward_margin" in metrics


def test_dpo_trainer_step_smoke():
    cfg = ModelConfig(
        vocab_size=64,
        max_seq_len=16,
        d_model=32,
        n_layers=1,
        n_heads=4,
        n_kv_heads=2,
        d_ff=64,
        dropout=0.0,
    )
    policy = JagXTransformer(cfg)
    step = DPOTrainerStep(policy, reference=None, config=DPOConfig(reference_free=True))
    chosen = torch.randint(0, 64, (2, 8))
    rejected = torch.randint(0, 64, (2, 8))
    loss, metrics = step(chosen, rejected)
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in policy.parameters() if p.grad is not None]
    assert len(grads) > 0
