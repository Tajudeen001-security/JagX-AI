import tempfile
from pathlib import Path

import torch
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

from api.local_model import LocalModelService
from api.server import create_app
from model import ModelConfig, JagXTransformer
from tokenizer import JagXTokenizer
from tokenizer.wrapper import SPECIAL_TOKENS


def _tok(tmpdir: Path) -> JagXTokenizer:
    corpus = tmpdir / "c.txt"
    corpus.write_text("hello jagx api test\n" * 20, encoding="utf-8")
    tok = Tokenizer(BPE(unk_token="<unk>"))
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
    trainer = BpeTrainer(vocab_size=120, special_tokens=SPECIAL_TOKENS, min_frequency=1)
    tok.train([str(corpus)], trainer)
    w = JagXTokenizer(tok)
    w.save(tmpdir / "tok")
    return JagXTokenizer.from_pretrained(tmpdir / "tok")


def test_local_model_service_generate():
    cfg = ModelConfig(
        vocab_size=120,
        max_seq_len=32,
        d_model=32,
        n_layers=1,
        n_heads=4,
        n_kv_heads=2,
        d_ff=64,
        dropout=0.0,
    )
    model = JagXTransformer(cfg)
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        tok = _tok(root)
        # Align vocab loosely for smoke: reload model with tok size if needed
        svc = LocalModelService(model, tok)
        out = svc.generate({"prompt": "hello", "max_tokens": 3, "temperature": 1.0, "top_k": 5})
        assert out["backend"] == "local-jagx"
        assert out["external_ai_api_required"] is False
        assert isinstance(out["text"], str)

        Handler = create_app(generate_fn=svc.as_generate_fn())
        api_out = Handler.post_routes["/v1/generate"]({"prompt": "hi", "max_tokens": 2})
        assert api_out["backend"] == "local-jagx"
