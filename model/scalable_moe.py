from __future__ import annotations

"""Scalable JagX architecture track.

This module is opt-in so existing checkpoints remain compatible. It adds a
Top-2 routed Mixture-of-Experts feed-forward layer and a longer-context model
configuration without changing the original JagXTransformer.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .transformer import CausalSelfAttention, RMSNorm, RotaryEmbedding


@dataclass
class ScalableModelConfig:
    vocab_size: int = 32768
    max_seq_len: int = 8192
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    n_kv_heads: int = 4
    d_ff: int = 2048
    num_experts: int = 8
    top_k: int = 2
    dropout: float = 0.0
    rope_theta: float = 500000.0
    rms_norm_eps: float = 1e-6
    tie_embeddings: bool = True
    router_aux_loss_coef: float = 0.01

    def validate(self) -> "ScalableModelConfig":
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if self.num_experts < 2 or not 1 <= self.top_k <= self.num_experts:
            raise ValueError("invalid MoE expert/top_k configuration")
        if self.max_seq_len < 2 or self.n_layers < 1:
            raise ValueError("invalid sequence/layer configuration")
        return self


class SwiGLUExpert(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class Top2MoE(nn.Module):
    """Token-level top-k router with a Switch-style load-balancing loss."""

    def __init__(self, cfg: ScalableModelConfig):
        super().__init__()
        self.num_experts = cfg.num_experts
        self.top_k = cfg.top_k
        self.router_aux_loss_coef = cfg.router_aux_loss_coef
        self.router = nn.Linear(cfg.d_model, cfg.num_experts, bias=False)
        self.experts = nn.ModuleList([
            SwiGLUExpert(cfg.d_model, cfg.d_ff, cfg.dropout)
            for _ in range(cfg.num_experts)
        ])
        self.last_aux_loss = torch.tensor(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        flat = x.reshape(-1, d)
        router_logits = self.router(flat)
        probs = F.softmax(router_logits, dim=-1)
        top_values, top_indices = torch.topk(probs, k=self.top_k, dim=-1)
        top_values = top_values / top_values.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        output = torch.zeros_like(flat)
        for expert_id, expert in enumerate(self.experts):
            token_positions, choice_positions = torch.where(top_indices == expert_id)
            if token_positions.numel() == 0:
                continue
            expert_out = expert(flat[token_positions])
            weights = top_values[token_positions, choice_positions].unsqueeze(-1)
            output.index_add_(0, token_positions, expert_out * weights)

        # Importance/load balancing: penalize routers that collapse onto a few experts.
        mean_prob = probs.mean(dim=0)
        hard_load = F.one_hot(top_indices, num_classes=self.num_experts).float().sum(dim=(0, 1))
        hard_load = hard_load / max(float(flat.size(0) * self.top_k), 1.0)
        aux = self.num_experts * torch.sum(mean_prob * hard_load)
        self.last_aux_loss = aux * self.router_aux_loss_coef
        return output.reshape(b, t, d)


class ScalableBlock(nn.Module):
    def __init__(self, cfg: ScalableModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.norm2 = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        # Reuse JagX's tested GQA attention implementation through a tiny adapter.
        class AttentionCfg:
            pass
        ac = AttentionCfg()
        ac.n_heads = cfg.n_heads
        ac.n_kv_heads = cfg.n_kv_heads
        ac.d_model = cfg.d_model
        ac.dropout = cfg.dropout
        self.attn = CausalSelfAttention(ac)
        self.moe = Top2MoE(cfg)

    def forward(self, x, cos, sin, past_kv=None, use_cache=False):
        h, present = self.attn(self.norm1(x), cos, sin, past_kv=past_kv, use_cache=use_cache)
        x = x + h
        x = x + self.moe(self.norm2(x))
        return x, present


class ScalableJagXTransformer(nn.Module):
    """JagX Transformer with GQA + Top-2 MoE + long-context-ready RoPE."""

    def __init__(self, cfg: ScalableModelConfig):
        super().__init__()
        self.cfg = cfg.validate()
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.rope = RotaryEmbedding(
            cfg.d_model // cfg.n_heads,
            max_seq_len=cfg.max_seq_len,
            theta=cfg.rope_theta,
        )
        self.blocks = nn.ModuleList([ScalableBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None,
                past_key_values: Optional[list] = None, use_cache: bool = False):
        b, t = input_ids.shape
        if t > self.cfg.max_seq_len and past_key_values is None:
            raise ValueError(f"Sequence length {t} exceeds max_seq_len {self.cfg.max_seq_len}")
        x = self.token_embedding(input_ids)
        past_len = 0
        if past_key_values and past_key_values[0] is not None:
            past_len = past_key_values[0][0].shape[2]
        cos, sin = self.rope(past_len + t)
        cos = cos[:, :, past_len:past_len + t, :]
        sin = sin[:, :, past_len:past_len + t, :]
        presents = [] if use_cache else None
        aux = x.new_zeros(())
        for i, block in enumerate(self.blocks):
            past = past_key_values[i] if past_key_values is not None else None
            x, present = block(x, cos, sin, past_kv=past, use_cache=use_cache)
            aux = aux + block.moe.last_aux_loss
            if use_cache:
                presents.append(present)
        logits = self.lm_head(self.norm(x))
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)) + aux
        if use_cache:
            return logits, loss, presents
        return logits, loss
