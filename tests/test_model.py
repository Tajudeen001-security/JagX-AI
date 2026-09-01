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
    logits, loss = m(x, labels=x)
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


def test_backward_and_update():
    c = ModelConfig(
        vocab_size=64,
        max_seq_len=16,
        d_model=32,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        d_ff=64,
    )
    m = JagXTransformer(c)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    x = torch.randint(0, 64, (2, 8))
    logits, loss = m(x, labels=x)
    assert loss is not None
    loss.backward()
    opt.step()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert len(grads) > 0


@torch.no_grad()
def test_kv_cache_equivalence():
    """Full-sequence forward logits on last position must match cached autoregressive step."""
    torch.manual_seed(0)
    c = ModelConfig(
        vocab_size=50,
        max_seq_len=32,
        d_model=32,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        d_ff=64,
        dropout=0.0,
        use_swiglu=True,
        use_rms_norm=True,
    )
    m = JagXTransformer(c)
    m.eval()

    prompt = torch.randint(0, 50, (2, 6))  # batch > 1
    full_logits, _ = m(prompt)
    full_last = full_logits[:, -1, :]

    past = None
    for i in range(prompt.size(1)):
        tok = prompt[:, i : i + 1]
        logits, _, past = m(tok, past_key_values=past, use_cache=True)
    cached_last = logits[:, -1, :]

    assert torch.allclose(full_last, cached_last, rtol=1e-4, atol=1e-5), (
        f"max diff={(full_last - cached_last).abs().max().item()}"
    )


@torch.no_grad()
def test_kv_cache_multi_step_generation_consistency():
    """Greedy generation with and without cache should produce identical tokens."""
    torch.manual_seed(42)
    c = ModelConfig(
        vocab_size=40,
        max_seq_len=64,
        d_model=32,
        n_layers=2,
        n_heads=4,
        n_kv_heads=4,
        d_ff=64,
        dropout=0.0,
    )
    m = JagXTransformer(c)
    m.eval()
    prompt = torch.randint(0, 40, (1, 4))

    def greedy_generate(use_cache: bool, steps: int = 5):
        past = None
        generated = prompt.clone()
        for _ in range(steps):
            if use_cache and past is not None:
                model_input = generated[:, -1:]
            else:
                model_input = generated
            if use_cache:
                logits, _, past = m(model_input, past_key_values=past, use_cache=True)
            else:
                logits, _ = m(model_input)
                past = None
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
        return generated

    out_cache = greedy_generate(True)
    out_full = greedy_generate(False)
    assert torch.equal(out_cache, out_full), f"cache={out_cache.tolist()} full={out_full.tolist()}"


def test_weight_tying():
    c = ModelConfig(vocab_size=32, max_seq_len=16, d_model=16, n_layers=1, n_heads=2, tie_embeddings=True)
    m = JagXTransformer(c)
    assert m.lm_head.weight is m.token_embedding.weight


def test_max_seq_len_enforced():
    c = ModelConfig(vocab_size=32, max_seq_len=8, d_model=16, n_layers=1, n_heads=2)
    m = JagXTransformer(c)
    x = torch.randint(0, 32, (1, 12))
    try:
        m(x)
        assert False, "should have raised"
    except ValueError:
        pass
