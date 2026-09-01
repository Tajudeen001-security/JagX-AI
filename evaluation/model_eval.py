from __future__ import annotations

from typing import Optional

import torch

from model import ModelConfig, JagXTransformer
from evaluation.runner import BenchmarkRunner, BenchmarkResult


def _greedy_continuation(model: JagXTransformer, prompt_ids: list[int], max_new: int = 8) -> list[int]:
    x = torch.tensor([prompt_ids], dtype=torch.long)
    with torch.no_grad():
        out = model.generate(x, max_new_tokens=max_new, temperature=0.01, top_k=1, top_p=1.0)
    return out[0].tolist()


def instruction_following_smoke(model: Optional[JagXTransformer] = None) -> BenchmarkResult:
    """Tiny deterministic check: model produces finite logits and continuations.

    This is infrastructure verification, not a claim of trained capability.
    """
    if model is None:
        cfg = ModelConfig(
            vocab_size=128,
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

    samples = [
        [1, 2, 3, 4],
        [5, 6, 7, 8, 9],
        [10, 11],
    ]

    def evaluator(prompt_ids: list[int]) -> float:
        cont = _greedy_continuation(model, prompt_ids, max_new=4)
        if len(cont) <= len(prompt_ids):
            return 0.0
        # Score 1.0 if continuation length is correct and ids in vocab
        ok_len = len(cont) == len(prompt_ids) + 4
        ok_ids = all(0 <= t < model.cfg.vocab_size for t in cont)
        return 1.0 if ok_len and ok_ids else 0.0

    runner = BenchmarkRunner(threshold=1.0)
    return runner.run("instruction_following_smoke", samples, evaluator)


def coding_shape_smoke(model: Optional[JagXTransformer] = None) -> BenchmarkResult:
    """Model forward+generate under coding-like short prompts."""
    if model is None:
        cfg = ModelConfig(
            vocab_size=96,
            max_seq_len=24,
            d_model=32,
            n_layers=1,
            n_heads=4,
            d_ff=64,
            dropout=0.0,
        )
        model = JagXTransformer(cfg)
        model.eval()

    samples = [[1, 2, 3], [4, 5, 6, 7]]

    def evaluator(ids: list[int]) -> float:
        x = torch.tensor([ids], dtype=torch.long)
        logits, loss = model(x, labels=x)
        if not torch.isfinite(logits).all():
            return 0.0
        if loss is not None and not torch.isfinite(loss):
            return 0.0
        return 1.0

    return BenchmarkRunner(threshold=1.0).run("coding_shape_smoke", samples, evaluator)
