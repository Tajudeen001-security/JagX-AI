from __future__ import annotations

import json
from pathlib import Path

import torch

from jagx.cli import main
from model import JagXTransformer, ModelConfig


ROOT = Path(__file__).resolve().parents[1]


def test_estimated_parameters_close_to_real_tiny_model():
    cfg = ModelConfig(
        vocab_size=128,
        max_seq_len=32,
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        d_ff=128,
        use_swiglu=True,
        use_rms_norm=True,
        tie_embeddings=True,
    )
    model = JagXTransformer(cfg)
    estimated = cfg.estimated_parameters()
    actual = model.parameter_count()
    assert estimated > 0
    assert abs(estimated - actual) / actual < 0.15


def test_kaggle_config_is_much_larger_than_e2e_toy():
    cfg = ModelConfig.from_dict(json.loads((ROOT / "configs" / "kaggle.json").read_text()))
    assert cfg.d_model >= 512
    assert cfg.n_layers >= 8
    assert cfg.vocab_size >= 16000
    assert cfg.estimated_parameters() > 20_000_000


def test_large_and_xlarge_configs_validate():
    for name in ("large.json", "xlarge.json", "kaggle_medium.json"):
        cfg = ModelConfig.from_dict(json.loads((ROOT / "configs" / name).read_text()))
        cfg.validate()
        assert cfg.estimated_parameters() > cfg.d_model * cfg.n_layers


def test_sdpa_and_manual_attention_match():
    torch.manual_seed(0)
    kwargs = dict(
        vocab_size=48,
        max_seq_len=32,
        d_model=32,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        d_ff=64,
        dropout=0.0,
    )
    x = torch.randint(0, 48, (2, 6))
    manual = JagXTransformer(ModelConfig(use_sdpa=False, **kwargs))
    sdpa = JagXTransformer(ModelConfig(use_sdpa=True, **kwargs))
    sdpa.load_state_dict(manual.state_dict())
    manual.eval()
    sdpa.eval()
    a, _ = manual(x)
    b, _ = sdpa(x)
    assert torch.allclose(a, b, rtol=1e-4, atol=1e-4)


def test_cli_inspect_kaggle_config():
    assert main(["inspect", "--config", str(ROOT / "configs" / "kaggle.json")]) == 0


def test_gaming_seed_exists_and_is_usable():
    path = ROOT / "data" / "seed" / "gaming_instructions.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) >= 15
    assert all(row["license"] == "MIT" and "Assistant:" in row["text"] for row in rows)
