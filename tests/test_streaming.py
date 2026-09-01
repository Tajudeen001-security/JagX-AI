import torch
from model import ModelConfig, JagXTransformer
from runtime.generation import GenerationConfig
from runtime.streaming import stream_generate


def test_stream_generate_yields_tokens():
    cfg = ModelConfig(
        vocab_size=64,
        max_seq_len=32,
        d_model=32,
        n_layers=1,
        n_heads=4,
        n_kv_heads=2,
        d_ff=64,
        dropout=0.0,
    )
    model = JagXTransformer(cfg)
    model.eval()
    x = torch.randint(0, 64, (1, 4))
    tokens = list(stream_generate(model, x, GenerationConfig(max_new_tokens=3, temperature=1.0, top_k=10)))
    assert len(tokens) == 3
    assert all(t.shape == (1, 1) for t in tokens)
