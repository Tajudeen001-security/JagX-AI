from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .config import ModelConfig


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    # q, k: (B, n_heads, T, head_dim)
    # cos/sin: (1, 1, T, head_dim)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., dim)
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 2048, theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)  # (T, dim/2)
        emb = torch.cat((freqs, freqs), dim=-1)  # (T, dim)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.cos_cached.shape[2]:
            self._build_cache(seq_len)
        return (
            self.cos_cached[:, :, :seq_len, :],
            self.sin_cached[:, :, :seq_len, :],
        )


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads  # for GQA

        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

        # Causal mask is built on the fly for flexibility with different lengths

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_kv_heads, T, head_dim) -> (B, n_heads, T, head_dim)
        if self.n_rep == 1:
            return x
        b, n_kv, t, d = x.shape
        x = x[:, :, None, :, :].expand(b, n_kv, self.n_rep, t, d)
        return x.reshape(b, n_kv * self.n_rep, t, d)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        b, t, _ = x.shape

        q = self.q_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE only to the new tokens' positions
        # When past_kv is present, cos/sin already cover the full sequence length
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        present = (k, v) if use_cache else None

        k = self._repeat_kv(k)
        v = self._repeat_kv(v)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, n_heads, T, T_total)

        # Causal mask
        t_total = scores.size(-1)
        causal = torch.tril(
            torch.ones(t, t_total, device=x.device, dtype=torch.bool),
            diagonal=t_total - t,
        )
        scores = scores.masked_fill(~causal[None, None, :, :], torch.finfo(scores.dtype).min)

        if attn_mask is not None:
            scores = scores + attn_mask

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)  # (B, n_heads, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(b, t, -1)
        return self.o_proj(out), present


class SwiGLUMLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.w1 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)  # gate
        self.w3 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)  # up
        self.w2 = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)  # down
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.fc1 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.fc2 = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        Norm = RMSNorm if cfg.use_rms_norm else nn.LayerNorm
        eps = cfg.rms_norm_eps if cfg.use_rms_norm else 1e-5
        self.norm1 = Norm(cfg.d_model, eps=eps) if cfg.use_rms_norm else Norm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = Norm(cfg.d_model, eps=eps) if cfg.use_rms_norm else Norm(cfg.d_model)
        self.mlp = SwiGLUMLP(cfg) if cfg.use_swiglu else MLP(cfg)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        h, present = self.attn(self.norm1(x), cos, sin, past_kv=past_kv, use_cache=use_cache)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x, present


class JagXTransformer(nn.Module):
    """Configurable causal Transformer for JagX.

    Features:
    - RoPE positional encoding
    - RMSNorm (or LayerNorm)
    - SwiGLU (or GELU MLP)
    - Grouped-query attention (GQA) via n_kv_heads
    - Optional weight tying
    - KV-cache support for efficient generation
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        cfg = cfg.validate()
        self.cfg = cfg

        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.rope = RotaryEmbedding(
            cfg.d_model // cfg.n_heads, max_seq_len=cfg.max_seq_len, theta=cfg.rope_theta
        )
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        Norm = RMSNorm if cfg.use_rms_norm else nn.LayerNorm
        eps = cfg.rms_norm_eps if cfg.use_rms_norm else 1e-5
        self.norm = Norm(cfg.d_model, eps=eps) if cfg.use_rms_norm else Norm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        if cfg.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.initializer_range)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[list] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[list]]:
        """
        Returns:
            logits, loss (optional), present_key_values (optional)
        """
        b, t = input_ids.shape
        if t > self.cfg.max_seq_len and past_key_values is None:
            raise ValueError(f"Sequence length {t} exceeds max_seq_len {self.cfg.max_seq_len}")

        x = self.token_embedding(input_ids)

        # Position offset when using cache
        past_len = 0
        if past_key_values is not None and len(past_key_values) > 0 and past_key_values[0] is not None:
            past_len = past_key_values[0][0].shape[2]

        cos, sin = self.rope(past_len + t)
        # Slice the rotary embeddings for the new tokens only
        cos = cos[:, :, past_len : past_len + t, :]
        sin = sin[:, :, past_len : past_len + t, :]

        presents = [] if use_cache else None
        for i, block in enumerate(self.blocks):
            past_kv = past_key_values[i] if past_key_values is not None else None
            x, present = block(x, cos, sin, past_kv=past_kv, use_cache=use_cache)
            if use_cache:
                presents.append(present)

        logits = self.lm_head(self.norm(x))

        loss = None
        if labels is not None:
            # Shift for causal LM: predict next token
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return logits, loss, presents

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 64,
        temperature: float = 0.8,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.95,
        repetition_penalty: float = 1.0,
        stop_token_ids: Optional[list[int]] = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """Autoregressive generation with temperature, top-k, top-p and repetition penalty."""
        self.eval()
        stop_set = set(stop_token_ids or [])
        past = None
        generated = input_ids

        for _ in range(max_new_tokens):
            if use_cache and past is not None:
                # Only feed the last token when cache is active
                model_input = generated[:, -1:]
            else:
                model_input = generated[:, -self.cfg.max_seq_len :]

            logits, _, past = self(model_input, past_key_values=past, use_cache=use_cache)
            logits = logits[:, -1, :]  # (B, vocab)

            # Temperature
            logits = logits / max(temperature, 1e-5)

            # Repetition penalty
            if repetition_penalty != 1.0:
                for b in range(generated.size(0)):
                    for token_id in set(generated[b].tolist()):
                        val = logits[b, token_id]
                        if val > 0:
                            logits[b, token_id] = val / repetition_penalty
                        else:
                            logits[b, token_id] = val * repetition_penalty

            # Top-k
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            # Top-p (nucleus)
            if top_p is not None and 0.0 < top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative > top_p
                # Keep at least the first token
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)

            if stop_set and any(int(next_token[b, 0]) in stop_set for b in range(next_token.size(0))):
                break

        return generated
