import pytest
import torch

from model import JagXTransformer, ModelConfig
from training.multimodal_data import load_multimodal_jsonl
from training.pretraining import PretrainingConfig
from training.unified_multimodal import MultimodalBatch, UnifiedMultimodalModel, train_unified


class TinyTokenizer:
    pad_token_id = 0
    vocab_size = 32

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [min(ord(c), 27) + 1 for c in text][:16]


def make_model():
    cfg = ModelConfig(vocab_size=32, max_seq_len=64, d_model=32, n_layers=2, n_heads=4)
    return UnifiedMultimodalModel(JagXTransformer(cfg), patch_size=4, audio_stride=4)


def test_unified_model_text_only_and_image():
    model = make_model()
    ids = torch.ones(2, 6, dtype=torch.long)
    labels = ids.clone()
    logits, loss = model(MultimodalBatch(ids, labels))
    assert logits.shape[:2] == ids.shape
    assert torch.isfinite(loss)
    images = torch.rand(2, 3, 8, 8)
    logits, loss = model(MultimodalBatch(ids, labels, images=images))
    assert logits.shape[0] == 2
    assert torch.isfinite(loss)


def test_unified_model_audio_and_video():
    model = make_model()
    ids = torch.ones(1, 4, dtype=torch.long)
    labels = ids.clone()
    audio = torch.rand(1, 1, 16)
    video = torch.rand(1, 3, 2, 8, 8)
    logits, loss = model(MultimodalBatch(ids, labels, audio=audio, video=video))
    assert logits.shape[0] == 1
    assert torch.isfinite(loss)


def test_unified_trainer_runs():
    model_config = ModelConfig(vocab_size=32, max_seq_len=64, d_model=32, n_layers=2, n_heads=4)
    batches = [
        MultimodalBatch(
            torch.ones(1, 4, dtype=torch.long),
            torch.ones(1, 4, dtype=torch.long),
            images=torch.rand(1, 3, 16, 16),
        )
    ]
    result = train_unified(
        batches, TinyTokenizer(), model_config,
        PretrainingConfig(seq_len=4, batch_size=1, max_steps=1, learning_rate=1e-3),
        device="cpu",
    )
    assert result["steps"] == 1
    assert result["mean_loss"] >= 0


def test_multimodal_data_requires_modality(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text("{\"text\":\"hello\"}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="at least one modality"):
        load_multimodal_jsonl(path)
