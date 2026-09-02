from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import torch
from torch import nn

from media.multimodal_model import JagXMultimodalModel
from model import JagXTransformer, ModelConfig
from tokenizer import JagXTokenizer
from training.pretraining import PretrainingConfig
from training.seed import set_seed


@dataclass
class MultimodalBatch:
    input_ids: torch.Tensor
    labels: torch.Tensor
    images: Optional[torch.Tensor] = None
    audio: Optional[torch.Tensor] = None
    video: Optional[torch.Tensor] = None


class UnifiedMultimodalModel(nn.Module):
    """Route one or more modalities into the JagX language model."""

    def __init__(self, language_model: JagXTransformer, patch_size: int = 16, audio_stride: int = 320):
        super().__init__()
        if patch_size <= 0 or audio_stride <= 0:
            raise ValueError("patch_size and audio_stride must be positive")
        self.language_model = language_model
        self.image = JagXMultimodalModel(language_model, patch_size=patch_size).image_encoder
        self.audio_proj = nn.Conv1d(1, language_model.cfg.d_model, audio_stride, audio_stride)
        self.audio_norm = nn.LayerNorm(language_model.cfg.d_model)
        self.video = nn.Conv3d(
            3, language_model.cfg.d_model,
            kernel_size=(1, patch_size, patch_size),
            stride=(1, patch_size, patch_size),
        )

    def _prefixes(self, batch: MultimodalBatch) -> list[torch.Tensor]:
        text_device = self.language_model.token_embedding(batch.input_ids).device
        prefixes = []
        if batch.images is not None:
            prefixes.append(self.image(batch.images.to(text_device)))
        if batch.audio is not None:
            audio = self.audio_proj(batch.audio.to(text_device)).transpose(1, 2)
            prefixes.append(self.audio_norm(audio))
        if batch.video is not None:
            v = batch.video.to(text_device)
            if v.ndim != 5:
                raise ValueError("video must have shape [B,C,T,H,W]")
            prefixes.append(self.video(v).flatten(2).transpose(1, 2))
        return prefixes

    def forward(self, batch: MultimodalBatch):
        text = self.language_model.token_embedding(batch.input_ids)
        prefixes = self._prefixes(batch)
        if not prefixes:
            return self.language_model(batch.input_ids, labels=batch.labels)
        if any(p.shape[0] != text.shape[0] for p in prefixes):
            raise ValueError("all modality batch sizes must match")
        embeddings = torch.cat([*prefixes, text], dim=1)
        if embeddings.shape[1] > self.language_model.cfg.max_seq_len:
            raise ValueError("combined modality prefixes exceed max_seq_len")
        prefix_len = embeddings.shape[1] - text.shape[1]
        prefix_labels = torch.full(
            (batch.labels.shape[0], prefix_len), -100,
            device=batch.labels.device, dtype=batch.labels.dtype,
        )
        labels = torch.cat([prefix_labels, batch.labels.to(prefix_labels.device)], dim=1)
        model = self.language_model
        cos, sin = model.rope(embeddings.shape[1])
        x = embeddings
        for block in model.blocks:
            x, _ = block(x, cos.to(x.device), sin.to(x.device))
        logits = model.lm_head(model.norm(x))
        loss = torch.nn.functional.cross_entropy(
            logits[..., :-1, :].contiguous().view(-1, logits.size(-1)),
            labels[..., 1:].contiguous().view(-1),
            ignore_index=-100,
        )
        return logits, loss


def train_unified(
    batches: Iterable[MultimodalBatch],
    tokenizer: JagXTokenizer,
    model_config: ModelConfig,
    config: PretrainingConfig,
    device: str | None = None,
) -> dict:
    del tokenizer
    set_seed(config.seed)
    model = UnifiedMultimodalModel(JagXTransformer(model_config))
    target = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(target)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    steps = 0
    total = 0.0
    for batch in batches:
        if steps >= config.max_steps:
            break
        moved = MultimodalBatch(
            input_ids=batch.input_ids.to(target),
            labels=batch.labels.to(target),
            images=None if batch.images is None else batch.images.to(target),
            audio=None if batch.audio is None else batch.audio.to(target),
            video=None if batch.video is None else batch.video.to(target),
        )
        _, loss = model(moved)
        if loss is None or not torch.isfinite(loss):
            raise FloatingPointError("non-finite multimodal loss")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        steps += 1
        total += float(loss.detach())
    if not steps:
        raise ValueError("no multimodal training steps were executed")
    return {"steps": steps, "mean_loss": total / steps}
