import argparse
import torch
from model import ModelConfig, JagXTransformer
from tokenizers import Tokenizer

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", required=True)
p.add_argument("--tokenizer", required=True)
p.add_argument("--prompt", required=True)
p.add_argument("--tokens", type=int, default=128)
p.add_argument("--temperature", type=float, default=0.8)
p.add_argument("--top-k", type=int, default=50)
p.add_argument("--top-p", type=float, default=0.95)
a = p.parse_args()

# Prefer weights_only for safety when loading trusted artifacts.
ckpt = torch.load(a.checkpoint, map_location="cpu", weights_only=True)
cfg_data = ckpt.get("config", {})
if isinstance(cfg_data, dict):
    cfg = ModelConfig.from_dict(cfg_data)
else:
    cfg = ModelConfig(**cfg_data) if hasattr(cfg_data, "__dict__") else ModelConfig()
model = JagXTransformer(cfg)
model.load_state_dict(ckpt["model"])
model.eval()

tok = Tokenizer.from_file(a.tokenizer)
x = torch.tensor([tok.encode(a.prompt).ids], dtype=torch.long)
y = model.generate(
    x,
    max_new_tokens=a.tokens,
    temperature=a.temperature,
    top_k=a.top_k,
    top_p=a.top_p,
)[0].tolist()
print(tok.decode(y))
