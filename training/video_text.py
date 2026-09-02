from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import torch
from torch import nn

from model import JagXTransformer, ModelConfig
from tokenizer import JagXTokenizer
from training.pretraining import PretrainingConfig
from training.seed import set_seed


class VideoFrameEncoder(nn.Module):
    """Native trainable video frame-to-token encoder."""

    def __init__(self, d_model: int, patch_size: int = 16, channels: int = 3):
        super().__init__()
        if d_model <= 0 or patch_size <= 0 or channels <= 0:
            raise ValueError("d_model, patch_size, and channels must be positive")
        self.patch_size = patch_size
        self.proj = nn.Conv3d(
            channels,
            d_model,
            kernel_size=(1, patch_size, patch_size),
            stride=(1, patch_size, patch_size),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        if video.ndim != 5:
            raise ValueError("video must have shape [B,C,T,H,W]")
        if video.shape[1] != self.proj.in_channels:
            raise ValueError(f"expected {self.proj.in_channels} video channels, got {video.shape[1]}")
        if video.shape[-2] < self.patch_size or video.shape[-1] < self.patch_size:
            raise ValueError("video height and width must be at least patch_size")
        h = (video.shape[-2] // self.patch_size) * self.patch_size
        w = (video.shape[-1] // self.patch_size) * self.patch_size
        video = video[..., :h, :w]
        tokens = self.proj(video).flatten(2).transpose(1, 2)
        return self.norm(tokens)


class JagXVideoConditioner(nn.Module):
    """Video-conditioned JagX language model with native temporal features."""

    def __init__(self, language_model: JagXTransformer, patch_size: int = 16):
        super().__init__()
        self.language_model = language_model
        self.video_encoder = VideoFrameEncoder(language_model.cfg.d_model, patch_size=patch_size)

    def forward(self, input_ids: torch.Tensor, video: torch.Tensor, labels: torch.Tensor | None = None):
        text = self.language_model.token_embedding(input_ids)
        video_tokens = self.video_encoder(video.to(text.device, dtype=text.dtype))
        if video_tokens.shape[0] != text.shape[0]:
            raise ValueError("video batch size must match input_ids batch size")
        total = video_tokens.shape[1] + text.shape[1]
        if total > self.language_model.cfg.max_seq_len:
            raise ValueError("video prefix plus text exceeds max_seq_len")
        embeddings = torch.cat([video_tokens, text], dim=1)
        full_labels = None
        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError("labels must match input_ids shape")
            prefix = torch.full((labels.shape[0], video_tokens.shape[1]), -100, device=labels.device, dtype=labels.dtype)
            full_labels = torch.cat([prefix, labels.to(prefix.device)], dim=1)
        model = self.language_model
        cos, sin = model.rope(embeddings.shape[1])
        x = embeddings
        for block in model.blocks:
            x, _ = block(x, cos.to(x.device), sin.to(x.device))
        logits = model.lm_head(model.norm(x))
        loss = None
        if full_labels is not None:
            loss = torch.nn.functional.cross_entropy(
                logits[..., :-1, :].contiguous().view(-1, logits.size(-1)),
                full_labels[..., 1:].contiguous().view(-1),
                ignore_index=-100,
            )
        return logits, loss


def load_video(path: str | Path) -> torch.Tensor:
    path = Path(path)
    if path.suffix.lower() in {".pt", ".pth"}:
        video = torch.load(path, map_location="cpu", weights_only=True)
    else:
        try:
            import imageio.v3 as iio
        except ImportError as exc:
            raise RuntimeError("imageio is required for non-tensor video files") from exc
        frames = iio.imread(path, index=None)
        video = torch.from_numpy(frames)
        if video.ndim == 4:
            video = video.permute(0, 3, 1, 2)
    video = video.float()
    if video.ndim != 4:
        raise ValueError("video must have shape [T,C,H,W] on disk")
    if video.shape[1] != 3:
        if video.shape[-1] == 3:
            video = video.permute(0, 3, 1, 2)
        else:
            raise ValueError("video must have 3 RGB channels")
    if video.max() > 1:
        video = video / 255.0
    return video.clamp(0, 1)


def load_paired_examples(path: str | Path) -> list[dict]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"paired data not found: {source}")
    rows = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not row.get("video") or not row.get("text"):
                raise ValueError(f"line {line_number} must contain video and text")
            rows.append(row)
    if not rows:
        raise ValueError("paired dataset is empty")
    return rows


def video_batches(rows: list[dict], tokenizer: JagXTokenizer, cfg: PretrainingConfig) -> Iterable[dict]:
    for start in range(0, len(rows), cfg.batch_size):
        chunk = rows[start : start + cfg.batch_size]
        if len(chunk) < cfg.batch_size and cfg.drop_remainder:
            break
        tokens = [tokenizer.encode(str(row["text"]), add_special_tokens=True)[: cfg.seq_len] for row in chunk]
        if any(not item for item in tokens):
            continue
        videos = [load_video(row["video"]) for row in chunk]
        if len({tuple(video.shape) for video in videos}) != 1:
            raise ValueError("all videos in a batch must have identical [T,C,H,W] dimensions")
        video_batch = torch.stack(videos).permute(0, 2, 1, 3, 4)
        input_ids = torch.full((len(chunk), cfg.seq_len), tokenizer.pad_token_id, dtype=torch.long)
        labels = torch.full_like(input_ids, -100)
        for index, item in enumerate(tokens):
            input_ids[index, : len(item)] = torch.tensor(item)
            labels[index, : len(item)] = input_ids[index, : len(item)]
        yield {"input_ids": input_ids, "video": video_batch, "labels": labels}


def train_video_text(
    data_path: str | Path,
    tokenizer_path: str | Path,
    model_config: ModelConfig,
    config: PretrainingConfig,
    output_dir: str | Path = "checkpoints/video",
    device: str | None = None,
) -> dict:
    set_seed(config.seed)
    tokenizer = JagXTokenizer.from_pretrained(tokenizer_path)
    model = JagXVideoConditioner(JagXTransformer(model_config))
    target = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(target)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    model.train()
    total_loss = 0.0
    steps = 0
    for batch in video_batches(load_paired_examples(data_path), tokenizer, config):
        if steps >= config.max_steps:
            break
        moved = {key: value.to(target) for key, value in batch.items()}
        _, loss = model(**moved)
        if loss is None or not torch.isfinite(loss):
            raise FloatingPointError("non-finite video-text loss")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        steps += 1
        total_loss += float(loss.detach())
    if not steps:
        raise ValueError("no video-text training steps were executed")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    checkpoint = Path(output_dir) / "video-text-final.pt"
    torch.save({"model": model.state_dict(), "step": steps, "mean_loss": total_loss / steps}, checkpoint)
    return {"steps": steps, "mean_loss": total_loss / steps, "checkpoint": str(checkpoint)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train JagX on paired video-text examples")
    parser.add_argument("--data", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default=None)
    parser.add_argument("--out-dir", default="checkpoints/video")
    args = parser.parse_args()
    tokenizer = JagXTokenizer.from_pretrained(args.tokenizer)
    model_config = ModelConfig(vocab_size=tokenizer.vocab_size, max_seq_len=2048, d_model=384, n_layers=6, n_heads=6)
    config = PretrainingConfig(seq_len=args.seq_len, batch_size=args.batch_size, max_steps=args.steps)
    print(json.dumps(train_video_text(args.data, args.tokenizer, model_config, config, args.out_dir, args.device), indent=2))


if __name__ == "__main__":
    main()
