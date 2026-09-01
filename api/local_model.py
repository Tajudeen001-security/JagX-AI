from __future__ import annotations

from typing import Any, Optional

import torch

from inference.loader import generate_text, load_model, load_tokenizer
from model import JagXTransformer
from tokenizer import JagXTokenizer


class LocalModelService:
    """Binds a loaded JagX model+tokenizer for API generate_fn use.

    Fully local; never calls external AI provider APIs.
    """

    def __init__(self, model: JagXTransformer, tokenizer: JagXTokenizer, device: Optional[str] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or next(model.parameters()).device

    @classmethod
    def from_paths(cls, checkpoint: str, tokenizer_path: str, device: Optional[str] = None) -> "LocalModelService":
        model, _ = load_model(checkpoint, device=device)
        tok = load_tokenizer(tokenizer_path)
        return cls(model, tok, device=device)

    def generate(self, payload: dict[str, Any]) -> dict[str, Any]:
        prompt = str(payload.get("prompt") or payload.get("input") or "")
        max_tokens = int(payload.get("max_tokens") or payload.get("tokens") or 64)
        temperature = float(payload.get("temperature") or 0.8)
        top_k = payload.get("top_k", 50)
        top_p = float(payload.get("top_p") or 0.95)
        text = generate_text(
            self.model,
            self.tokenizer,
            prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=int(top_k) if top_k is not None else None,
            top_p=top_p,
            device=str(self.device),
        )
        return {
            "object": "jagx.generation",
            "backend": "local-jagx",
            "external_ai_api_required": False,
            "prompt_chars": len(prompt),
            "max_tokens": max_tokens,
            "text": text,
        }

    def as_generate_fn(self):
        return self.generate
