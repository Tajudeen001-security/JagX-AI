import torch

from training.pretraining import PretrainingConfig, packed_batches, prepare_examples
from training.trainer import CausalLMTrainer, TrainerConfig
from training.data_contract import TrainingExample


class FakeTokenizer:
    bos_token_id = 2
    eos_token_id = 3
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [ord(char) % 20 + 4 for char in text]


def test_prepare_examples_filters_and_deduplicates():
    examples = [
        TrainingExample("JagX builds useful software for everyone.", "a"),
        TrainingExample("JagX builds useful software for everyone.", "b"),
        TrainingExample("A second training document with enough content.", "c"),
    ]
    processed, stats = prepare_examples(examples, seed=7)
    assert len(processed) == 2
    assert stats.rejected_duplicate == 1


def test_packed_batches_fixed_shape_and_labels():
    tokenizer = FakeTokenizer()
    examples = [TrainingExample("abcdefghij", "unit-test")]
    cfg = PretrainingConfig(seq_len=4, batch_size=2, drop_remainder=False)
    batches = list(packed_batches(examples, tokenizer, cfg))
    assert len(batches) == 1
    assert batches[0]["input_ids"].shape == (2, 4)
    assert batches[0]["labels"].shape == (2, 4)
    assert torch.equal(batches[0]["labels"][0], batches[0]["input_ids"][0])
    assert (batches[0]["labels"][1] == -100).any()


def test_trainer_moves_batches_to_device_and_learns():
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.2)

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([[0.0]]))

        def forward(self, input_ids, labels):
            del labels
            x = input_ids.float().unsqueeze(-1)
            prediction = x * self.weight
            return ((prediction - 1.0) ** 2).mean()

    tiny = TinyModel()
    trainer = CausalLMTrainer(
        tiny,
        optimizer=torch.optim.SGD(tiny.parameters(), lr=0.2),
        config=TrainerConfig(max_steps=2, grad_accum=1, use_amp=False, device="cpu", save_every=100),
    )
    batch = {"input_ids": torch.ones(2, 2, dtype=torch.long), "labels": torch.ones(2, 2, dtype=torch.long)}
    result = trainer.train([batch])
    assert result
    assert tiny.weight.item() > 0
