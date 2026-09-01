import torch
from model import ModelConfig, JagXTransformer


def test_forward_shape():
    c = ModelConfig(
        vocab_size=128,
        max_seq_len=32,
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        d_ff=128,
        use_swiglu=True,
        use_rms_norm=True,
    )
    m = JagXTransformer(c)
    x = torch.randint(0, 128, (2, 16))
    logits, loss, _ = m(x, labels=x)
    assert logits.shape == (2, 16, 128)
    assert loss is not None and loss.ndim == 0


def test_generation_shape():
    c = ModelConfig(
        vocab_size=64,
        max_seq_len=32,
        d_model=32,
        n_layers=1,
        n_heads=4,
        n_kv_heads=4,
        d_ff=64,
        use_swiglu=False,
        use_rms_norm=False,
    )
    m = JagXTransformer(c)
    x = torch.randint(0, 64, (1, 4))
    out = m.generate(x, max_new_tokens=3, temperature=0.9, top_k=10, top_p=0.9)
    assert out.shape == (1, 7)


def test_gqa_and_rope():
    c = ModelConfig(
        vocab_size=100,
        max_seq_len=64,
        d_model=64,
        n_layers=2,
        n_heads=8,
        n_kv_heads=2,
        use_swiglu=True,
        use_rms_norm=True,
    )
    m = JagXTransformer(c)
    x = torch.randint(0, 100, (1, 8))
    logits, loss, presents = m(x, labels=x, use_cache=True)
    assert logits.shape == (1, 8, 100)
    assert loss is not None
    assert presents is not None and len(presents) == 2
    # KV cache shapes: (B, n_kv_heads, T, head_dim)
    assert presents[0][0].shape == (1, 2, 8, 8)


def test_config_from_dict():
    d = {
        "vocab_size": 256,
        "max_seq_len": 128,
        "d_model": 48,
        "n_layers": 2,
        "n_heads": 4,
        "n_kv_heads": 2,
    }
    c = ModelConfig.from_dict(d)
    assert c.d_model == 48
    assert c.n_kv_heads == 2
    m = JagXTransformer(c)
    assert m.cfg.d_ff > 0
