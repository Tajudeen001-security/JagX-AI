from __future__ import annotations


def pack_tokens(tokens: list[int], seq_len: int, drop_remainder: bool = True) -> list[list[int]]:
    """Pack a flat token stream into fixed-length causal-LM sequences."""
    if seq_len < 2:
        raise ValueError("seq_len must be at least 2")
    if any(not isinstance(t, int) or t < 0 for t in tokens):
        raise ValueError("tokens must be non-negative integers")
    limit = len(tokens) - (len(tokens) % seq_len) if drop_remainder else len(tokens)
    return [
        tokens[i : i + seq_len]
        for i in range(0, limit, seq_len)
        if len(tokens[i : i + seq_len]) == seq_len or not drop_remainder
    ]


def causal_pairs(sequence: list[int]) -> tuple[list[int], list[int]]:
    if len(sequence) < 2:
        raise ValueError("sequence must contain at least two tokens")
    return sequence[:-1], sequence[1:]
