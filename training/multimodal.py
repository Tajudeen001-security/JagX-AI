from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import torch
from torch import nn

from media.multimodal_model import JagXMultimodalModel
from model import JagXTransformer, ModelConfig
from tokenizer import JagXTokenizer
from training.pretraining import PretrainingConfig
from training.seed import set_seed


def load_paired_examples(path: str | Path) -> list[dict]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"paired data not found: {source}")
    rows = []
    with source.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not row.get("image") or not row.get("text"):
                raise ValueError(f"line {number} must contain image and text")
            rows.append(row)
    if not rows:
        raise ValueError("paired dataset is empty")
    return rows


def load_image(path: str | Path) -> torch.Tensor:
    image_path = Path(path)
    if image_path.suffix.lower() in {".pt", ".pth"}:
        image = torch.load(image_path, map_location="cpu", weights_only=True)
    else:
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("Pillow is required for non-tensor image files") from exc
        import numpy as np
        image = torch.from_numpy(np.asarray(Image.open(image_path).convert("RGB"))).permute(2, 0, 1)
    image = image.float()
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("image must have shape [3,H,W]")
    if image.max() > 1:
        image = image / 255.0
    return image.clamp(0, 1)


def paired_batches(rows: list[dict], tokenizer: JagXTokenizer, cfg: PretrainingConfig) -> Iterable[dict]:
    prepared = []
    for row in rows:
        tokens = tokenizer.encode(str(row["text"]), add_special_tokens=True)[: cfg.seq_len]
        if tokens:
            prepared.append((row, tokens))
    if not prepared:
        raise ValueError("paired dataset produced no tokenized examples")
    for start in range(0, len(prepared), cfg.batch_size):
        chunk = prepared[start : start + cfg.batch_size]
        if len(chunk) < cfg.batch_size and cfg.drop_remainder:
            break
        input_ids = torch.full((len(chunk), cfg.seq_len), tokenizer.pad_token_id, dtype=torch.long)
        labels = torch.full_like(input_ids, -100)
        images = []
        for index, (row, tokens) in enumerate(chunk):
            input_ids[index, : len(tokens)] = torch.tensor(tokens, dtype=torch.long)
            labels[index, : len(tokens)] = input_ids[index, : len(tokens)]
            images.append(load_image(row["image"]))
        if len({tuple(image.shape) for image in images}) != 1:
            raise ValueError("all images in a batch must have identical dimensions")
        yield {"input_ids": input_ids, "images": torch.stack(images), "labels": labels}


def train_multimodal(
    data_path: str | Path,
    tokenizer_path: str | Path,
    model_config: ModelConfig,
    config: PretrainingConfig,
    output_dir: str | Path = "checkpoints/multimodal",
    device: str | None = None,
) -> dict:
    set_seed(config.seed)
    tokenizer = JagXTokenizer.from_pretrained(tokenizer_path)
    model = JagXMultimodalModel(JagXTransformer(model_config))
    target = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(target)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    model.train()
    step = 0
    total_loss = 0.0
    for batch in paired_batches(load_paired_examples(data_path), tokenizer, config):
        if step >= config.max_steps:
            break
        moved = {key: value.to(target) for key, value in batch.items()}
        _, loss = model(**moved)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite multimodal loss")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        total_loss += float(loss.detach())
    if step == 0:
        raise ValueError("no multimodal training steps were executed")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    checkpoint = Path(output_dir) / "multimodal-final.pt"
    torch.save({"model": model.state_dict(), "step": step, "mean_loss": total_loss / step}, checkpoint)
    return {"steps": step, "mean_loss": total_loss / step, "checkpoint": str(checkpoint)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train JagX on paired image-text examples")
    parser.add_argument("--data", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default=None)
    parser.add_argument("--out-dir", default="checkpoints/multimodal")
    args = parser.parse_args()
    tokenizer = JagXTokenizer.from_pretrained(args.tokenizer)
    model_config = ModelConfig(vocab_size=tokenizer.vocab_size, max_seq_len=2048, d_model=384, n_layers=6, n_heads=6)
    config = PretrainingConfig(seq_len=args.seq_len, batch_size=args.batch_size, max_steps=args.steps)
    print(json.dumps(train_multimodal(args.data, args.tokenizer, model_config, config, args.out_dir, args.device), indent=2))


if __name__ == "__main__":
    main()
