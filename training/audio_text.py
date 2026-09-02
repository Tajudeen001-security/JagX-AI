from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import torch
from torch import nn

from model import JagXTransformer, ModelConfig
from tokenizer import JagXTokenizer
from training.data_contract import TrainingExample
from training.pretraining import PretrainingConfig
from training.seed import set_seed


class AudioFrameEncoder(nn.Module):
    """Native trainable audio waveform-to-token encoder."""

    def __init__(self, d_model: int, channels: int = 1, stride: int = 320):
        super().__init__()
        if d_model <= 0 or channels <= 0 or stride <= 0:
            raise ValueError("d_model, channels, and stride must be positive")
        self.channels = channels
        self.stride = stride
        self.proj = nn.Conv1d(channels, d_model, kernel_size=stride, stride=stride)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.ndim != 3:
            raise ValueError("audio must have shape [B,C,S]")
        if audio.shape[1] != self.channels:
            raise ValueError(f"expected {self.channels} audio channels, got {audio.shape[1]}")
        if audio.shape[-1] < self.stride:
            raise ValueError("audio sequence must be at least one stride long")
        frames = self.proj(audio)
        return self.norm(frames.transpose(1, 2))


class JagXAudioConditioner(nn.Module):
    """Audio-conditioned JagX language model using native trainable features."""

    def __init__(self, language_model: JagXTransformer, stride: int = 320):
        super().__init__()
        self.language_model = language_model
        self.audio_encoder = AudioFrameEncoder(language_model.cfg.d_model, stride=stride)

    def forward(self, input_ids: torch.Tensor, audio: torch.Tensor, labels: torch.Tensor | None = None):
        text = self.language_model.token_embedding(input_ids)
        audio_tokens = self.audio_encoder(audio.to(text.device, dtype=text.dtype))
        if audio_tokens.shape[0] != text.shape[0]:
            raise ValueError("audio batch size must match input_ids batch size")
        total = audio_tokens.shape[1] + text.shape[1]
        if total > self.language_model.cfg.max_seq_len:
            raise ValueError("audio prefix plus text exceeds max_seq_len")
        embeddings = torch.cat([audio_tokens, text], dim=1)
        full_labels = None
        if labels is not None:
            prefix = torch.full((labels.shape[0], audio_tokens.shape[1]), -100, device=labels.device, dtype=labels.dtype)
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


def load_audio(path: str | Path) -> torch.Tensor:
    path = Path(path)
    if path.suffix.lower() in {".pt", ".pth"}:
        audio = torch.load(path, map_location="cpu", weights_only=True)
    else:
        try:
            import soundfile as sf
        except ImportError as exc:
            raise RuntimeError("soundfile is required for non-tensor audio files") from exc
        data, _ = sf.read(path, always_2d=True)
        audio = torch.from_numpy(data).transpose(0, 1)
    audio = audio.float()
    if audio.ndim != 2:
        raise ValueError("audio must have shape [C,S] on disk")
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    return audio


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
            if not isinstance(row, dict) or not row.get("audio") or not row.get("text"):
                raise ValueError(f"line {line_number} must contain audio and text")
            rows.append(row)
    if not rows:
        raise ValueError("paired dataset is empty")
    return rows


def audio_batches(rows: list[dict], tokenizer: JagXTokenizer, cfg: PretrainingConfig) -> Iterable[dict]:
    for start in range(0, len(rows), cfg.batch_size):
        chunk = rows[start : start + cfg.batch_size]
        if len(chunk) < cfg.batch_size and cfg.drop_remainder:
            break
        tokens = [tokenizer.encode(str(row["text"]), add_special_tokens=True)[: cfg.seq_len] for row in chunk]
        if any(not item for item in tokens):
            continue
        max_samples = max(load_audio(row["audio"]).shape[-1] for row in chunk)
        stride = 320
        max_samples = ((max_samples + stride - 1) // stride) * stride
        audio_batch = torch.zeros(len(chunk), 1, max_samples)
        input_ids = torch.full((len(chunk), cfg.seq_len), tokenizer.pad_token_id, dtype=torch.long)
        labels = torch.full_like(input_ids, -100)
        for index, (row, item) in enumerate(zip(chunk, tokens)):
            audio = load_audio(row["audio"])
            audio_batch[index, :, : audio.shape[-1]] = audio
            input_ids[index, : len(item)] = torch.tensor(item)
            labels[index, : len(item)] = input_ids[index, : len(item)]
        yield {"input_ids": input_ids, "audio": audio_batch, "labels": labels}


def train_audio_text(
    data_path: str | Path,
    tokenizer_path: str | Path,
    model_config: ModelConfig,
    config: PretrainingConfig,
    output_dir: str | Path = "checkpoints/audio",
    device: str | None = None,
) -> dict:
    set_seed(config.seed)
    tokenizer = JagXTokenizer.from_pretrained(tokenizer_path)
    model = JagXAudioConditioner(JagXTransformer(model_config))
    target = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(target)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    model.train()
    total_loss = 0.0
    steps = 0
    for batch in audio_batches(load_paired_examples(data_path), tokenizer, config):
        if steps >= config.max_steps:
            break
        moved = {key: value.to(target) for key, value in batch.items()}
        _, loss = model(**moved)
        if loss is None or not torch.isfinite(loss):
            raise FloatingPointError("non-finite audio-text loss")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        steps += 1
        total_loss += float(loss.detach())
    if not steps:
        raise ValueError("no audio-text training steps were executed")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    checkpoint = Path(output_dir) / "audio-text-final.pt"
    torch.save({"model": model.state_dict(), "step": steps, "mean_loss": total_loss / steps}, checkpoint)
    return {"steps": steps, "mean_loss": total_loss / steps, "checkpoint": str(checkpoint)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train JagX on paired audio-text examples")
    parser.add_argument("--data", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default=None)
    parser.add_argument("--out-dir", default="checkpoints/audio")
    args = parser.parse_args()
    tokenizer = JagXTokenizer.from_pretrained(args.tokenizer)
    model_config = ModelConfig(vocab_size=tokenizer.vocab_size, max_seq_len=2048, d_model=384, n_layers=6, n_heads=6)
    config = PretrainingConfig(seq_len=args.seq_len, batch_size=args.batch_size, max_steps=args.steps)
    print(json.dumps(train_audio_text(args.data, args.tokenizer, model_config, config, args.out_dir, args.device), indent=2))


if __name__ == "__main__":
    main()
