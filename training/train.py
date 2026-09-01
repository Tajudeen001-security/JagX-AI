import argparse
from pathlib import Path
import torch
from model import ModelConfig,JagXTransformer

def train(args):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg=ModelConfig(vocab_size=args.vocab_size,max_seq_len=args.seq_len,d_model=args.d_model,n_layers=args.layers,n_heads=args.heads,d_ff=args.ff)
    model=JagXTransformer(cfg).to(device); opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.1)
    tokens=torch.tensor(list(map(int,Path(args.tokens).read_text().split())),dtype=torch.long)
    model.train()
    for step in range(args.steps):
        span=args.batch_size*args.seq_len; start=(step*span)%max(1,len(tokens)-span-1)
        x=tokens[start:start+span].view(args.batch_size,args.seq_len).to(device); y=tokens[start+1:start+1+span].view(args.batch_size,args.seq_len).to(device)
        _,loss=model(x,y); opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        if step%args.log_every==0: print(f"step={step} loss={loss.item():.4f}")
    Path(args.out).parent.mkdir(parents=True,exist_ok=True); torch.save({"model":model.state_dict(),"config":cfg.__dict__},args.out)

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--tokens",required=True); p.add_argument("--out",default="checkpoints/jagx.pt"); p.add_argument("--vocab-size",type=int,default=32000); p.add_argument("--seq-len",type=int,default=512); p.add_argument("--d-model",type=int,default=512); p.add_argument("--layers",type=int,default=8); p.add_argument("--heads",type=int,default=8); p.add_argument("--ff",type=int,default=2048); p.add_argument("--batch-size",type=int,default=2); p.add_argument("--steps",type=int,default=1000); p.add_argument("--lr",type=float,default=3e-4); p.add_argument("--log-every",type=int,default=10); train(p.parse_args())
