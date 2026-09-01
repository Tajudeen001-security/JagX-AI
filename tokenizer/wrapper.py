from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Sequence, Union

from tokenizers import Tokenizer
from tokenizers.processors import TemplateProcessing


SPECIAL_TOKENS = [
    "<pad>",
    "<unk>",
    "<bos>",
    "<eos>",
    "<tool>",
    "<file>",
    "<code>",
    "<image>",
    "<audio>",
]


class JagXTokenizer:
    """Runtime wrapper around a trained HuggingFace tokenizers BPE model.

    Provides a stable encode/decode API, special-token helpers, attention masks,
    and deterministic serialization so training and inference stay in sync.
    """

    VERSION = "1.0.0"

    def __init__(self, tokenizer: Tokenizer, metadata: Optional[dict] = None):
        self._tok = tokenizer
        self.metadata = metadata or {}
        self._ensure_special_ids()

    def _ensure_special_ids(self) -> None:
        vocab = self._tok.get_vocab()
        self.pad_token_id = vocab.get("<pad>", 0)
        self.unk_token_id = vocab.get("<unk>", 1)
        self.bos_token_id = vocab.get("<bos>", 2)
        self.eos_token_id = vocab.get("<eos>", 3)

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "JagXTokenizer":
        path = Path(path)
        tok = Tokenizer.from_file(str(path))
        meta_path = path.parent / "metadata.json"
        metadata = {}
        if meta_path.exists():
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        return cls(tok, metadata)

    @classmethod
    def from_pretrained(cls, directory: Union[str, Path]) -> "JagXTokenizer":
        directory = Path(directory)
        return cls.from_file(directory / "tokenizer.json")

    def save(self, directory: Union[str, Path]) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self._tok.save(str(directory / "tokenizer.json"))
        meta = {
            "version": self.VERSION,
            "vocab_size": self.vocab_size,
            "special_tokens": SPECIAL_TOKENS,
            "pad_token_id": self.pad_token_id,
            "unk_token_id": self.unk_token_id,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
            **self.metadata,
        }
        (directory / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        truncation: bool = True,
    ) -> list[int]:
        ids = self._tok.encode(text).ids
        if add_special_tokens:
            ids = [self.bos_token_id] + ids + [self.eos_token_id]
        if max_length is not None and truncation and len(ids) > max_length:
            ids = ids[: max_length - 1] + [self.eos_token_id]
        return ids

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        if skip_special_tokens:
            special = {
                self.pad_token_id,
                self.unk_token_id,
                self.bos_token_id,
                self.eos_token_id,
            }
            ids = [i for i in ids if i not in special]
        return self._tok.decode(list(ids))

    def batch_encode(
        self,
        texts: Sequence[str],
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        padding: bool = True,
        truncation: bool = True,
        return_attention_mask: bool = True,
    ) -> dict[str, Any]:
        encoded = [
            self.encode(t, add_special_tokens=add_special_tokens, max_length=max_length, truncation=truncation)
            for t in texts
        ]
        if max_length is None:
            max_length = max(len(e) for e in encoded) if encoded else 0

        input_ids = []
        attention_mask = []
        for e in encoded:
            if truncation and len(e) > max_length:
                e = e[:max_length]
            pad_len = max_length - len(e) if padding else 0
            input_ids.append(e + [self.pad_token_id] * pad_len)
            if return_attention_mask:
                attention_mask.append([1] * len(e) + [0] * pad_len)

        result: dict[str, Any] = {"input_ids": input_ids}
        if return_attention_mask:
            result["attention_mask"] = attention_mask
        return result

    def enable_bos_eos_template(self) -> None:
        """Optionally force BOS/EOS via post-processor (for training scripts)."""
        self._tok.post_processor = TemplateProcessing(
            single="<bos> $A <eos>",
            special_tokens=[
                ("<bos>", self.bos_token_id),
                ("<eos>", self.eos_token_id),
            ],
        )
