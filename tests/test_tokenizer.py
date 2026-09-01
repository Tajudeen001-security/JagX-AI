from __future__ import annotations

import tempfile
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

from tokenizer import JagXTokenizer
from tokenizer.wrapper import SPECIAL_TOKENS


def _make_tiny_tokenizer(tmpdir: Path) -> JagXTokenizer:
    corpus = tmpdir / "corpus.txt"
    corpus.write_text("hello world\nhello jagx\njagx ai system\n" * 20, encoding="utf-8")
    tok = Tokenizer(BPE(unk_token="<unk>"))
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
    trainer = BpeTrainer(vocab_size=200, special_tokens=SPECIAL_TOKENS, min_frequency=1)
    tok.train([str(corpus)], trainer)
    wrapper = JagXTokenizer(tok)
    wrapper.save(tmpdir / "tok")
    return JagXTokenizer.from_pretrained(tmpdir / "tok")


def test_encode_decode_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        tok = _make_tiny_tokenizer(tmp)
        text = "hello jagx"
        ids = tok.encode(text)
        assert ids[0] == tok.bos_token_id
        assert ids[-1] == tok.eos_token_id
        decoded = tok.decode(ids)
        assert "hello" in decoded.lower() or "jagx" in decoded.lower()


def test_batch_encode_padding():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        tok = _make_tiny_tokenizer(tmp)
        batch = tok.batch_encode(["hi", "hello world jagx"], max_length=16, padding=True)
        assert len(batch["input_ids"]) == 2
        assert len(batch["input_ids"][0]) == 16
        assert len(batch["attention_mask"][0]) == 16
        assert sum(batch["attention_mask"][0]) <= 16


def test_special_token_ids_stable():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        tok = _make_tiny_tokenizer(tmp)
        assert tok.pad_token_id is not None
        assert tok.eos_token_id is not None
        assert tok.vocab_size >= len(SPECIAL_TOKENS)
