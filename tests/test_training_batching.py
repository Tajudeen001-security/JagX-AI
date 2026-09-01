import torch
from training.batching import iter_token_batches, sequences_to_tensors
from training.packing import pack_tokens


def test_pack_and_batch():
    tokens = list(range(40))
    seqs = pack_tokens(tokens, seq_len=8)
    assert all(len(s) == 8 for s in seqs)
    x, y = sequences_to_tensors(seqs[:2])
    assert x.shape == (2, 8)
    assert torch.equal(x, y)

    batches = list(iter_token_batches(tokens, seq_len=8, batch_size=2))
    assert len(batches) >= 1
    assert batches[0]["input_ids"].shape[0] == 2
