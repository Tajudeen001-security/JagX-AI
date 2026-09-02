from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

from model import ModelConfig, JagXTransformer
from tokenizer import JagXTokenizer
from tokenizer.wrapper import SPECIAL_TOKENS
from inference.loader import generate_text, load_model, load_tokenizer


def _tiny_tokenizer(tmpdir: Path) -> JagXTokenizer:
    corpus = tmpdir / "c.txt"
    corpus.write_text("hello jagx world\n" * 30, encoding="utf-8")
    tok = Tokenizer(BPE(unk_token="<unk>"))
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
    trainer = BpeTrainer(vocab_size=150, special_tokens=SPECIAL_TOKENS, min_frequency=1)
    tok.train([str(corpus)], trainer)
    wrapper = JagXTokenizer(tok)
    wrapper.save(tmpdir / "tok")
    return JagXTokenizer.from_pretrained(tmpdir / "tok")


def test_inference_loader_and_generate():
    cfg = ModelConfig(vocab_size=150, max_seq_len=32, d_model=32, n_layers=1, n_heads=4, n_kv_heads=2, d_ff=64, dropout=0.0)
    model = JagXTransformer(cfg)
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        ckpt = root / "m.pt"
        torch.save({"model": model.state_dict(), "config": cfg.to_dict()}, ckpt)
        _tiny_tokenizer(root)
        loaded, loaded_cfg = load_model(ckpt)
        assert loaded_cfg.d_model == 32
        loaded_tok = load_tokenizer(root / "tok")
        text = generate_text(loaded, loaded_tok, "hello", max_new_tokens=4, temperature=1.0, top_k=10)
        assert isinstance(text, str)


def test_inference_loader_accepts_metadata_model_config():
    cfg = ModelConfig(vocab_size=32, max_seq_len=16, d_model=16, n_layers=1, n_heads=4, d_ff=32, dropout=0.0)
    model = JagXTransformer(cfg)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "legacy.pt"
        torch.save({"model": model.state_dict(), "metadata": {"model_config": cfg.to_dict()}}, path)
        loaded, loaded_cfg = load_model(path)
        assert loaded_cfg.to_dict() == cfg.to_dict()
        for original, restored in zip(model.parameters(), loaded.parameters()):
            assert torch.equal(original, restored)


def test_inference_loader_rejects_unconfigured_incompatible_checkpoint():
    cfg = ModelConfig(vocab_size=32, max_seq_len=16, d_model=16, n_layers=1, n_heads=4, d_ff=32, dropout=0.0)
    model = JagXTransformer(cfg)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "unconfigured.pt"
        torch.save({"model": model.state_dict()}, path)
        with pytest.raises(ValueError, match="missing model config"):
            load_model(path)
