from __future__ import annotations

import tempfile
from pathlib import Path

from training.e2e_tiny import run_e2e


def test_e2e_tiny_train_and_generate():
    with tempfile.TemporaryDirectory() as td:
        out = run_e2e(td, steps=5, vocab_size=256, prompt="Training saves")
        assert Path(out["checkpoint"]).is_file()
        assert Path(out["tokenizer_dir"]).is_dir()
        assert out["backend"] == "local-jagx"
        assert isinstance(out["generated"], str)
        # Generation may be noisy on 5 steps; must still return a string from real decode
        assert out["external_ai_api_required"] is False
