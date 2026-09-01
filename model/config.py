from dataclasses import dataclass

@dataclass
class ModelConfig:
    vocab_size: int = 32000
    max_seq_len: int = 2048
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    d_ff: int = 2048
    dropout: float = 0.0
    tie_embeddings: bool = True

    def validate(self):
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.vocab_size < 2 or self.max_seq_len < 2:
            raise ValueError("vocab_size and max_seq_len must be >= 2")
