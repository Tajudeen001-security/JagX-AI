from __future__ import annotations

import math
from typing import Optional, Tuple, Union

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
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
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
        self.n_rep = self.n_heads // self.n_kv_heads

        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)
        self.use_sdpa = bool(getattr(cfg, "use_sdpa", True))

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
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
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        b, t, _ = x.shape

        q = self.q_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        present = (k, v) if use_cache else None

        k = self._repeat_kv(k)
        v = self._repeat_kv(v)

        dropout_p = self.dropout.p if self.training else 0.0
        can_sdpa = self.use_sdpa and hasattr(F, "scaled_dot_product_attention")
        if can_sdpa:
            t_total = k.size(2)
            if t == t_total:
                out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=True)
            else:
                causal = torch.ones(t, t_total, device=x.device, dtype=torch.bool).tril(diagonal=t_total - t)
                out = F.scaled_dot_product_attention(q, k, v, attn_mask=causal, dropout_p=dropout_p)
        else:
            scale = 1.0 / math.sqrt(self.head_dim)
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            t_total = scores.size(-1)
            causal = torch.tril(
                torch.ones(t, t_total, device=x.device, dtype=torch.bool),
                diagonal=t_total - t,
            )
            scores = scores.masked_fill(~causal[None, None, :, :], torch.finfo(scores.dtype).min)
            attn = F.softmax(scores, dim=-1)
            attn = self.dropout(attn)
            out = torch.matmul(attn, v)

        out = out.transpose(1, 2).contiguous().view(b, t, -1)
        return self.o_proj(out), present


class SwiGLUMLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.w1 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w3 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w2 = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)
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
        if cfg.use_rms_norm:
            self.norm1 = RMSNorm(cfg.d_model, eps=cfg.rms_norm_eps)
            self.norm2 = RMSNorm(cfg.d_model, eps=cfg.rms_norm_eps)
        else:
            self.norm1 = nn.LayerNorm(cfg.d_model)
            self.norm2 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
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
        self.rope = RotaryEmbedding(cfg.d_model // cfg.n_heads, max_seq_len=cfg.max_seq_len, theta=cfg.rope_theta)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        if cfg.use_rms_norm:
            self.norm = RMSNorm(cfg.d_model, eps=cfg.rms_norm_eps)
        else:
            self.norm = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        if cfg.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

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
    ) -> Union[
        Tuple[torch.Tensor, Optional[torch.Tensor]], Tuple[torch.Tensor, Optional[torch.Tensor], Optional[list]]
    ]:
        """
        Returns:
            When use_cache=False (default): (logits, loss)
            When use_cache=True: (logits, loss, present_key_values)
        This keeps existing training/eval code working while supporting KV cache.
        """
        b, t = input_ids.shape
        if t > self.cfg.max_seq_len and past_key_values is None:
            raise ValueError(f"Sequence length {t} exceeds max_seq_len {self.cfg.max_seq_len}")

        x = self.token_embedding(input_ids)

        past_len = 0
        if past_key_values is not None and len(past_key_values) > 0 and past_key_values[0] is not None:
            past_len = past_key_values[0][0].shape[2]

        cos, sin = self.rope(past_len + t)
        cos = cos[:, :, past_len : past_len + t, :]
        sin = sin[:, :, past_len : past_len + t, :]

        presents = [] if use_cache else None
        checkpoint = bool(self.cfg.gradient_checkpointing and self.training and not use_cache)
        for i, block in enumerate(self.blocks):
            past_kv = past_key_values[i] if past_key_values is not None else None
            if checkpoint:
                x, present = torch.utils.checkpoint.checkpoint(
                    block, x, cos, sin, past_kv, False, use_reentrant=False
                )
            else:
                x, present = block(x, cos, sin, past_kv=past_kv, use_cache=use_cache)
            if use_cache:
                presents.append(present)

        logits = self.lm_head(self.norm(x))

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        if use_cache:
            return logits, loss, presents
        return logits, loss

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
        self.eval()
        stop_set = set(stop_token_ids or [])
        past = None
        generated = input_ids

        for _ in range(max_new_tokens):
            if use_cache and past is not None:
                model_input = generated[:, -1:]
            else:
                model_input = generated[:, -self.cfg.max_seq_len :]

            if use_cache:
                logits, _, past = self(model_input, past_key_values=past, use_cache=True)
            else:
                logits, _ = self(model_input)
                past = None

            logits = logits[:, -1, :] / max(temperature, 1e-5)

            if repetition_penalty != 1.0:
                for b in range(generated.size(0)):
                    for token_id in set(generated[b].tolist()):
                        val = logits[b, token_id]
                        if val > 0:
                            logits[b, token_id] = val / repetition_penalty
                        else:
                            logits[b, token_id] = val * repetition_penalty

            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            if top_p is not None and 0.0 < top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative > top_p
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
