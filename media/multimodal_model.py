from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from model import JagXTransformer


class ImagePatchEncoder(nn.Module):
    """Trainable image-to-token encoder with no external model dependency."""

    def __init__(self, d_model: int, patch_size: int = 16, channels: int = 3):
        super().__init__()
        if d_model <= 0 or patch_size <= 0 or channels <= 0:
            raise ValueError("d_model, patch_size, and channels must be positive")
        self.patch_size = patch_size
        self.proj = nn.Conv2d(channels, d_model, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4:
            raise ValueError("images must have shape [B,C,H,W]")
        if images.shape[1] != self.proj.in_channels:
            raise ValueError(f"expected {self.proj.in_channels} image channels, got {images.shape[1]}")
        if images.shape[2] < self.patch_size or images.shape[3] < self.patch_size:
            raise ValueError("image height and width must be at least patch_size")
        height = (images.shape[2] // self.patch_size) * self.patch_size
        width = (images.shape[3] // self.patch_size) * self.patch_size
        if height != images.shape[2] or width != images.shape[3]:
            images = images[..., :height, :width]
        patches = self.proj(images)
        return self.norm(patches.flatten(2).transpose(1, 2))


class JagXMultimodalModel(nn.Module):
    """Native image-conditioned JagX language model.

    Image patches are projected into the same representation space as text and
    prepended as a soft prefix. Text labels are shifted so image prefix tokens
    never contribute to language-model loss.
    """

    def __init__(self, language_model: JagXTransformer, patch_size: int = 16):
        super().__init__()
        self.language_model = language_model
        self.image_encoder = ImagePatchEncoder(language_model.cfg.d_model, patch_size=patch_size)

    @property
    def cfg(self):
        return self.language_model.cfg

    def _image_prefix(self, images: torch.Tensor) -> torch.Tensor:
        return self.image_encoder(images)

    def forward(
        self,
        input_ids: torch.Tensor,
        images: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [B,T]")
        text_embeddings = self.language_model.token_embedding(input_ids)
        if images is None:
            return self.language_model(input_ids, labels=labels)
        image_embeddings = self._image_prefix(images.to(text_embeddings.device, dtype=text_embeddings.dtype))
        if image_embeddings.shape[0] != text_embeddings.shape[0]:
            raise ValueError("image batch size must match input_ids batch size")
        total_len = image_embeddings.shape[1] + text_embeddings.shape[1]
        if total_len > self.cfg.max_seq_len:
            raise ValueError("image prefix plus text exceeds max_seq_len")
        embeddings = torch.cat([image_embeddings, text_embeddings], dim=1)
        full_labels = None
        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError("labels must have the same shape as input_ids")
            prefix_labels = torch.full(
                (labels.shape[0], image_embeddings.shape[1]), -100, device=labels.device, dtype=labels.dtype
            )
            full_labels = torch.cat([prefix_labels, labels.to(prefix_labels.device)], dim=1)
        return self._forward_embeddings(embeddings, labels=full_labels)

    def _forward_embeddings(self, embeddings: torch.Tensor, labels: Optional[torch.Tensor] = None):
        model = self.language_model
        b, t, _ = embeddings.shape
        if t > model.cfg.max_seq_len:
            raise ValueError(f"Sequence length {t} exceeds max_seq_len {model.cfg.max_seq_len}")
        cos, sin = model.rope(t)
        cos = cos[:, :, :t, :].to(embeddings.device)
        sin = sin[:, :, :t, :].to(embeddings.device)
        x = embeddings
        for block in model.blocks:
            x, _ = block(x, cos, sin)
        logits = model.lm_head(model.norm(x))
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return logits, loss

    @torch.no_grad()
    def generate_with_image(
        self,
        input_ids: torch.Tensor,
        images: torch.Tensor,
        max_new_tokens: int = 64,
        temperature: float = 0.8,
        top_k: Optional[int] = 50,
    ) -> torch.Tensor:
        self.eval()
        text_embeddings = self.language_model.token_embedding(input_ids)
        image_embeddings = self._image_prefix(images.to(text_embeddings.device, dtype=text_embeddings.dtype))
        if image_embeddings.shape[0] != input_ids.shape[0]:
            raise ValueError("image batch size must match input_ids batch size")
        prefix = torch.cat([image_embeddings, text_embeddings], dim=1)
        if prefix.shape[1] >= self.cfg.max_seq_len:
            raise ValueError("prompt leaves no room for generated tokens")
        logits, _ = self._forward_embeddings(prefix)
        generated = input_ids
        past = None
        # Seed the base transformer's cache from the full multimodal prefix.
        _, _, past = self._forward_cached_embeddings(prefix)
        for _ in range(max_new_tokens):
            token = generated[:, -1:]
            logits, _, past = self.language_model(token, past_key_values=past, use_cache=True)
            scores = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k is not None and top_k > 0:
                values, _ = torch.topk(scores, min(top_k, scores.size(-1)))
                scores[scores < values[:, [-1]]] = float("-inf")
            next_token = torch.multinomial(F.softmax(scores, dim=-1), 1)
            generated = torch.cat([generated, next_token], dim=1)
        return generated

    def _forward_cached_embeddings(self, embeddings: torch.Tensor):
        model = self.language_model
        b, t, _ = embeddings.shape
        del b
        cos, sin = model.rope(t)
        cos = cos[:, :, :t, :].to(embeddings.device)
        sin = sin[:, :, :t, :].to(embeddings.device)
        x = embeddings
        presents = []
        for block in model.blocks:
            x, present = block(x, cos, sin, use_cache=True)
            presents.append(present)
        logits = model.lm_head(model.norm(x))
        return logits, None, presents
