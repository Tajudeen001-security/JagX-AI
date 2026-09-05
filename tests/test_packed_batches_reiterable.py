import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from tokenizer import JagXTokenizer
from training.data_contract import TrainingExample
from training.pretraining import PretrainingConfig, packed_batches


def make_tokenizer() -> JagXTokenizer:
    tokenizer = Tokenizer(WordLevel({"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3, "hello": 4, "world": 5}, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    return JagXTokenizer(tokenizer)


def test_packed_batches_can_be_restarted_after_exhaustion():
    tokenizer = make_tokenizer()
    examples = [TrainingExample(text="hello world hello world", source="test")]
    cfg = PretrainingConfig(seq_len=4, batch_size=1, drop_remainder=True)
    batches = packed_batches(examples, tokenizer, cfg)

    first = list(batches)
    second = list(batches)

    assert len(first) == 1
    assert len(second) == 1
    assert torch.equal(first[0]["input_ids"], second[0]["input_ids"])
