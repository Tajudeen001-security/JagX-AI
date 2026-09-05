import torch

from model.scalable_moe import ScalableJagXTransformer, ScalableModelConfig


def test_scalable_moe_forward_and_aux_loss():
    cfg = ScalableModelConfig(
        vocab_size=128,
        max_seq_len=64,
        d_model=96,
        n_layers=2,
        n_heads=6,
        n_kv_heads=2,
        d_ff=256,
        num_experts=4,
        top_k=2,
    )
    model = ScalableJagXTransformer(cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, 16))
    logits, loss = model(ids, labels=ids)
    assert logits.shape == (2, 16, cfg.vocab_size)
    assert loss is not None
    assert torch.isfinite(loss)
