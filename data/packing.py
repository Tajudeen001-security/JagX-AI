from __future__ import annotations

def pack_token_sequences(token_ids: list[int], seq_len: int) -> list[list[int]]:
    if seq_len < 2: raise ValueError("seq_len must be >= 2")
    usable=(len(token_ids)//seq_len)*seq_len
    return [token_ids[i:i+seq_len] for i in range(0,usable,seq_len)]
