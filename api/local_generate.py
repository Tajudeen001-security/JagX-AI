from __future__ import annotations

from typing import Any, Callable, Optional

from inference.loader import generate_text, load_model, load_tokenizer
from model import JagXTransformer
from tokenizer import JagXTokenizer


def make_generate_fn(
    model: JagXTransformer,
    tokenizer: JagXTokenizer,
) -> Callable[[dict], dict]:
    """Bind a loaded local JagX model+tokenizer into an API generate_fn.

    No external AI provider is used.
    """

    def generate_fn(payload: dict[str, Any]) -> dict[str, Any]:
        prompt = str(payload.get("prompt") or payload.get("input") or "")
        max_tokens = int(payload.get("max_tokens") or payload.get("tokens") or 64)
        temperature = float(payload.get("temperature") or 0.8)
        top_k = payload.get("top_k", 50)
        top_p = float(payload.get("top_p") or 0.95)
        repetition_penalty = float(payload.get("repetition_penalty") or 1.0)

        text = generate_text(
            model,
            tokenizer,
            prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=int(top_k) if top_k is not None else None,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
        return {
            "object": "jagx.generation",
            "backend": "local-jagx",
            "external_ai_api_required": False,
            "prompt_chars": len(prompt),
            "max_tokens": max_tokens,
            "text": text,
        }

    return generate_fn


def make_generate_fn_from_paths(
    checkpoint_path: str,
    tokenizer_path: str,
    device: Optional[str] = None,
) -> Callable[[dict], dict]:
    model, _ = load_model(checkpoint_path, device=device)
    tokenizer = load_tokenizer(tokenizer_path)
    return make_generate_fn(model, tokenizer)
