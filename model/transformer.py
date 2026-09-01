import math
import torch
from torch import nn
from .config import ModelConfig

class CausalSelfAttention(nn.Module):
    def __init__(self,cfg):
        super().__init__(); self.n_heads=cfg.n_heads; self.head_dim=cfg.d_model//cfg.n_heads
        self.qkv=nn.Linear(cfg.d_model,3*cfg.d_model,bias=False); self.out=nn.Linear(cfg.d_model,cfg.d_model,bias=False); self.dropout=nn.Dropout(cfg.dropout)
        self.register_buffer("mask",torch.tril(torch.ones(cfg.max_seq_len,cfg.max_seq_len,dtype=torch.bool)).view(1,1,cfg.max_seq_len,cfg.max_seq_len),persistent=False)
    def forward(self,x):
        b,t,c=x.shape; q,k,v=self.qkv(x).chunk(3,dim=-1)
        q=q.view(b,t,self.n_heads,self.head_dim).transpose(1,2); k=k.view(b,t,self.n_heads,self.head_dim).transpose(1,2); v=v.view(b,t,self.n_heads,self.head_dim).transpose(1,2)
        s=(q@k.transpose(-2,-1))/math.sqrt(self.head_dim); s=s.masked_fill(~self.mask[:,:,:t,:t],torch.finfo(s.dtype).min)
        a=self.dropout(torch.softmax(s,dim=-1)); return self.out((a@v).transpose(1,2).contiguous().view(b,t,c))

class Block(nn.Module):
    def __init__(self,cfg):
        super().__init__(); self.n1=nn.LayerNorm(cfg.d_model); self.attn=CausalSelfAttention(cfg); self.n2=nn.LayerNorm(cfg.d_model)
        self.mlp=nn.Sequential(nn.Linear(cfg.d_model,cfg.d_ff,bias=False),nn.GELU(),nn.Linear(cfg.d_ff,cfg.d_model,bias=False))
    def forward(self,x): return x+self.mlp(self.n2(x))+self.attn(self.n1(x))

class JagXTransformer(nn.Module):
    def __init__(self,cfg:ModelConfig):
        super().__init__(); cfg.validate(); self.cfg=cfg
        self.token_embedding=nn.Embedding(cfg.vocab_size,cfg.d_model); self.position_embedding=nn.Embedding(cfg.max_seq_len,cfg.d_model)
        self.blocks=nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)]); self.norm=nn.LayerNorm(cfg.d_model); self.lm_head=nn.Linear(cfg.d_model,cfg.vocab_size,bias=False)
        if cfg.tie_embeddings: self.lm_head.weight=self.token_embedding.weight
    def forward(self,input_ids,labels=None):
        _,t=input_ids.shape
        if t>self.cfg.max_seq_len: raise ValueError("Sequence exceeds max_seq_len")
        p=torch.arange(t,device=input_ids.device); x=self.token_embedding(input_ids)+self.position_embedding(p)[None]
        for block in self.blocks: x=block(x)
        logits=self.lm_head(self.norm(x)); loss=None
        if labels is not None: loss=nn.functional.cross_entropy(logits.reshape(-1,logits.size(-1)),labels.reshape(-1),ignore_index=-100)
        return logits,loss
    @torch.no_grad()
    def generate(self,input_ids,max_new_tokens=64,temperature=0.8,top_k=50):
        self.eval()
        for _ in range(max_new_tokens):
            logits,_=self(input_ids[:,-self.cfg.max_seq_len:]); logits=logits[:,-1,:]/max(temperature,1e-5)
            if top_k:
                values,_=torch.topk(logits,min(top_k,logits.size(-1))); logits[logits<values[:,-1,None]]=float('-inf')
            input_ids=torch.cat([input_ids,torch.multinomial(torch.softmax(logits,dim=-1),1)],dim=1)
        return input_ids
