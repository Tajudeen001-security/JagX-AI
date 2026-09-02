from __future__ import annotations

import pytest
import torch

from media.cached_generation import CachedGenerationConfig, generate_multimodal_cached
from model import JagXTransformer, ModelConfig
from training.unified_multimodal import MultimodalBatch, UnifiedMultimodalModel


def _model() -> UnifiedMultimodalModel:
    cfg = ModelConfig(
        vocab_size=32,
        max_seq_len=32,
        d_model=16,
        n_layers=1,
        n_heads=4,
        d_ff=32,
        dropout=0.0,
    )
    return UnifiedMultimodalModel(JagXTransformer(cfg), patch_size=4, audio_stride=4)


def _batch(modality: str | None) -> MultimodalBatch:
    ids = torch.tensor([[2, 4, 5]], dtype=torch.long)
    kwargs = {}
    if modality == "image":
        kwargs["images"] = torch.ones(1, 3, 8, 8)
    elif modality == "audio":
        kwargs["audio"] = torch.ones(1, 1, 8)
    elif modality == "video":
        kwargs["video"] = torch.ones(1, 3, 2, 8, 8)
    return MultimodalBatch(ids, ids.clone(), **kwargs)


def test_cached_generation_supports_all_modalities_and_text_only():
    torch.manual_seed(7)
    model = _model()
    for modality in (None, "image", "audio", "video"):
        output = generate_multimodal_cached(
            model,
            _batch(modality),
            CachedGenerationConfig(max_new_tokens=3, temperature=0),
        )
        assert output.shape == (1, 6)
        assert torch.isfinite(output.float()).all()


def test_cached_generation_is_greedy_and_does_not_duplicate_prompt_token():
    torch.manual_seed(11)
    model = _model()
    batch = _batch("image")
    output = generate_multimodal_cached(
        model, batch, CachedGenerationConfig(max_new_tokens=2, temperature=0)
    )
    assert torch.equal(output[:, : batch.input_ids.size(1)], batch.input_ids)
    assert output.size(1) == batch.input_ids.size(1) + 2


def test_cached_generation_rejects_prefix_that_leaves_no_room():
    cfg = ModelConfig(vocab_size=32, max_seq_len=8, d_model=16, n_layers=1, n_heads=4, d_ff=32)
    model = UnifiedMultimodalModel(JagXTransformer(cfg), patch_size=4, audio_stride=4)
    with pytest.raises(ValueError, match="insufficient max_seq_len"):
        generate_multimodal_cached(
            model,
            MultimodalBatch(torch.ones(1, 4, dtype=torch.long), torch.ones(1, 4, dtype=torch.long), images=torch.ones(1, 3, 8, 8)),
            CachedGenerationConfig(max_new_tokens=2, temperature=0),
        )
