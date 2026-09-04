from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from model import ModelConfig, JagXTransformer
from tokenizer import JagXTokenizer


def load_model(
    checkpoint_path: str | Path,
    device: Optional[str] = None,
    weights_only: bool = True,
) -> tuple[JagXTransformer, ModelConfig]:
    """Load a JagX checkpoint and reconstruct its exact model configuration."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    path = Path(checkpoint_path)
    ckpt = torch.load(path, map_location=device, weights_only=weights_only)
    if not isinstance(ckpt, dict) or "model" not in ckpt:
        raise ValueError("checkpoint must be a dict containing a 'model' state_dict")

    cfg_data = ckpt.get("config")
    if not isinstance(cfg_data, dict):
        metadata = ckpt.get("metadata")
        if isinstance(metadata, dict):
            cfg_data = metadata.get("model_config")
    if not isinstance(cfg_data, dict):
        cfg_data = {}

    cfg = ModelConfig.from_dict(cfg_data)
    try:
        model = JagXTransformer(cfg)
        model.load_state_dict(ckpt["model"])
    except (RuntimeError, ValueError) as exc:
        if not cfg_data:
            raise ValueError(
                "checkpoint is missing model config; re-save it with training.checkpoint.save_checkpoint "
                "or provide a checkpoint containing a 'config' field"
            ) from exc
        raise
    model.to(device)
    model.eval()
    return model, cfg


def load_tokenizer(tokenizer_path: str | Path) -> JagXTokenizer:
    path = Path(tokenizer_path)
    if path.is_dir():
        return JagXTokenizer.from_pretrained(path)
    return JagXTokenizer.from_file(path)


@torch.no_grad()
def generate_text(
    model: JagXTransformer,
    tokenizer: JagXTokenizer,
    prompt: str,
    max_new_tokens: int = 64,
    temperature: float = 0.8,
    top_k: Optional[int] = 50,
    top_p: float = 0.95,
    repetition_penalty: float = 1.0,
    stop_token_ids: Optional[list[int]] = None,
    device: Optional[str] = None,
    seed: Optional[int] = None,
) -> str:
    if seed is not None:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    device = device or next(model.parameters()).device
    ids = tokenizer.encode(prompt, add_special_tokens=True)
    if ids and ids[-1] == tokenizer.eos_token_id:
        ids = ids[:-1]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    stop = stop_token_ids or [tokenizer.eos_token_id]
    out = model.generate(
        x,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        stop_token_ids=stop,
    )
    return tokenizer.decode(out[0].tolist())
