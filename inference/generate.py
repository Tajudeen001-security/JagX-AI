import argparse,torch
from model import ModelConfig,JagXTransformer
from tokenizers import Tokenizer

p=argparse.ArgumentParser(); p.add_argument("--checkpoint",required=True); p.add_argument("--tokenizer",required=True); p.add_argument("--prompt",required=True); p.add_argument("--tokens",type=int,default=128); a=p.parse_args()
ckpt=torch.load(a.checkpoint,map_location="cpu"); model=JagXTransformer(ModelConfig(**ckpt["config"])); model.load_state_dict(ckpt["model"]); tok=Tokenizer.from_file(a.tokenizer)
x=torch.tensor([tok.encode(a.prompt).ids],dtype=torch.long); y=model.generate(x,a.tokens)[0].tolist(); print(tok.decode(y))
