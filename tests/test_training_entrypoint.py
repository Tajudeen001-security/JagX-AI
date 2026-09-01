from __future__ import annotations

import json
from pathlib import Path
import tempfile

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from model import ModelConfig
from tokenizer import JagXTokenizer
from training.entrypoint import build_model, evaluate_loss, load_examples, run_training
from training.pretraining import PretrainingConfig


def _tokenizer(directory: Path) -> JagXTokenizer:
    vocab = {
        "<pad>": 0,
        "<unk>": 1,
        "<bos>": 2,
        "<eos>": 3,
        "JagX": 4,
        "builds": 5,
        "useful": 6,
        "software": 7,
    }
    tok = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tok.pre_tokenizer = Whitespace()
    wrapper = JagXTokenizer(tok)
    wrapper.save(directory)
    return wrapper


def test_load_examples_validates_jsonl():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "data.jsonl"
        path.write_text(
            json.dumps({"text": "JagX builds useful software", "source": "unit", "split": "train"}) + "\n",
            encoding="utf-8",
        )
        examples = load_examples(path)
        assert len(examples) == 1
        assert examples[0].source == "unit"


def test_build_model_rejects_vocab_mismatch():
    with tempfile.TemporaryDirectory() as directory:
        tokenizer = _tokenizer(Path(directory))
        with pytest.raises(ValueError, match="does not match tokenizer"):
            build_model(ModelConfig(vocab_size=99, max_seq_len=8, d_model=16, n_layers=1, n_heads=4, d_ff=32), tokenizer)


def test_run_training_trains_and_reports_validation_loss():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        tokenizer = _tokenizer(root / "tokenizer")
        data = root / "data.jsonl"
        rows = [
            {"text": "JagX builds useful software", "source": "unit", "split": "train"},
            {"text": "JagX builds software", "source": "unit2", "split": "train"},
            {"text": "JagX useful software", "source": "val", "split": "validation"},
        ]
        data.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
        cfg = ModelConfig(vocab_size=tokenizer.vocab_size, max_seq_len=8, d_model=16, n_layers=1, n_heads=4, d_ff=32, dropout=0.0)
        train_cfg = PretrainingConfig(seq_len=4, batch_size=1, max_steps=1, grad_accum=1, learning_rate=1e-3, drop_remainder=False)
        result = run_training(data, root / "tokenizer", cfg, train_cfg, output_dir=root / "checkpoints", device="cpu")
        assert result["step"] == 1
        assert torch.isfinite(torch.tensor(result["train"]["mean_loss"]))
        assert torch.isfinite(torch.tensor(result["validation_loss"]))


def test_evaluate_loss_requires_batches():
    model = torch.nn.Linear(2, 2)
    with pytest.raises(ValueError, match="produced no batches"):
        evaluate_loss(model, [], torch.device("cpu"))
