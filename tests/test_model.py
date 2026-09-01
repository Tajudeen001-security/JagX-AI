import torch
from model import ModelConfig,JagXTransformer

def test_forward_shape():
    c=ModelConfig(vocab_size=128,max_seq_len=32,d_model=64,n_layers=2,n_heads=4,d_ff=128); m=JagXTransformer(c); x=torch.randint(0,128,(2,16)); logits,loss=m(x,x); assert logits.shape==(2,16,128); assert loss.ndim==0

def test_generation_shape():
    c=ModelConfig(vocab_size=64,max_seq_len=16,d_model=32,n_layers=1,n_heads=4,d_ff=64); m=JagXTransformer(c); x=torch.randint(0,64,(1,4)); assert m.generate(x,3).shape==(1,7)
