import torch
from model import ModelConfig, JagXTransformer
from training.dpo import DPOConfig, dpo_loss


def test_dpo_loss_finite():
    cfg = ModelConfig(
        vocab_size=32,
        max_seq_len=16,
        d_model=16,
        n_layers=1,
        n_heads=2,
        d_ff=32,
        dropout=0.0,
    )
    policy = JagXTransformer(cfg)
    ref = JagXTransformer(cfg)
    ref.load_state_dict(policy.state_dict())
    for p in ref.parameters():
        p.requires_grad_(False)

    chosen = torch.randint(0, 32, (2, 8))
    rejected = torch.randint(0, 32, (2, 8))

    with torch.no_grad():
        ref_c, _ = ref(chosen)
        ref_r, _ = ref(rejected)
    pol_c, _ = policy(chosen)
    pol_r, _ = policy(rejected)

    loss = dpo_loss(pol_c, pol_r, ref_c, ref_r, chosen, rejected, DPOConfig(beta=0.1))
    assert torch.isfinite(loss)
    loss.backward()
