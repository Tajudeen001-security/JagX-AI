from __future__ import annotations

from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

from .wrapper import SPECIAL_TOKENS, JagXTokenizer


def train(
    input_files: list[str],
    output_dir: str | Path,
    vocab_size: int = 32768,
    min_frequency: int = 2,
) -> JagXTokenizer:
    """Train JagX's BPE tokenizer from an explicit, pre-approved corpus list.

    Does not download data. Caller must supply only licensed / open files.
    """
    if not input_files:
        raise ValueError("at least one input file is required")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    tok = Tokenizer(BPE(unk_token="<unk>"))
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
    )
    tok.train(input_files, trainer)

    wrapper = JagXTokenizer(tok, metadata={"source_files": list(input_files), "min_frequency": min_frequency})
    wrapper.save(out)
    return wrapper


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Train a JagX BPE tokenizer")
    p.add_argument("--input", action="append", required=True, help="Text files used for training")
    p.add_argument("--output", default="artifacts/tokenizer")
    p.add_argument("--vocab-size", type=int, default=32768)
    p.add_argument("--min-frequency", type=int, default=2)
    a = p.parse_args()
    train(a.input, a.output, a.vocab_size, a.min_frequency)
    print(f"Tokenizer saved to {a.output}")
