from __future__ import annotations

import torch
import pytest

from media.multimodal_model import ImagePatchEncoder, JagXMultimodalModel
from model import JagXTransformer, ModelConfig


def _model() -> JagXMultimodalModel:
    cfg = ModelConfig(
        vocab_size=32,
        max_seq_len=64,
        d_model=16,
        n_layers=1,
        n_heads=4,
        n_kv_heads=4,
        d_ff=32,
        dropout=0.0,
    )
    return JagXMultimodalModel(JagXTransformer(cfg), patch_size=4)


def test_image_patch_encoder_returns_token_sequence():
    encoder = ImagePatchEncoder(16, patch_size=4)
    images = torch.randn(2, 3, 10, 12)
    tokens = encoder(images)
    assert tokens.shape == (2, 6, 16)


def test_multimodal_forward_masks_image_prefix_from_loss():
    model = _model()
    input_ids = torch.randint(0, 32, (2, 5))
    labels = input_ids.clone()
    images = torch.randn(2, 3, 16, 16)
    logits, loss = model(input_ids, images=images, labels=labels)
    assert logits.shape == (2, 21, 32)
    assert loss is not None and torch.isfinite(loss)


def test_multimodal_rejects_context_overflow():
    model = _model()
    input_ids = torch.randint(0, 32, (1, 64))
    images = torch.randn(1, 3, 16, 16)
    with pytest.raises(ValueError, match="exceeds max_seq_len"):
        model(input_ids, images=images)


def test_text_only_path_matches_base_model_shape():
    model = _model()
    input_ids = torch.randint(0, 32, (2, 5))
    logits, loss = model(input_ids)
    assert logits.shape == (2, 5, 32)
    assert loss is None
