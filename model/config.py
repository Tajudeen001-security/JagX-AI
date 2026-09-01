from dataclasses import dataclass

@dataclass
class ModelConfig:
    vocab_size:int=32768
    max_seq_len:int=2048
    d_model:int=384
    n_layers:int=6
    n_heads:int=6
    d_ff:int=1536
    dropout:float=0.0
    tie_embeddings:bool=True

    def validate(self):
        if self.d_model % self.n_heads: raise ValueError('d_model must be divisible by n_heads')
        if self.vocab_size<2 or self.max_seq_len<2: raise ValueError('vocab_size and max_seq_len must be >= 2')

    @classmethod
    def from_scale(cls,scale):
        return cls(vocab_size=scale.vocab,max_seq_len=scale.context,d_model=scale.d_model,n_layers=scale.layers,n_heads=scale.heads,d_ff=scale.d_model*4)
