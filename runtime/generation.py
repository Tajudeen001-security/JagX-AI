from __future__ import annotations
from dataclasses import dataclass
import torch

@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens:int=256
    temperature:float=0.8
    top_k:int|None=50
    top_p:float=0.95
    repetition_penalty:float=1.0

@torch.no_grad()
def generate(model,input_ids,config:GenerationConfig):
    model.eval(); ids=input_ids
    for _ in range(config.max_new_tokens):
        context=ids[:,-model.cfg.max_seq_len:]
        logits,_=model(context)
        logits=logits[:,-1,:]/max(config.temperature,1e-5)
        if config.repetition_penalty!=1.0:
            for token in ids[0].unique():
                value=logits[0,int(token)]
                logits[0,int(token)] = value/config.repetition_penalty if value>0 else value*config.repetition_penalty
        if config.top_k:
            v,_=torch.topk(logits,min(config.top_k,logits.size(-1))); logits[logits<v[:,-1:]]=float('-inf')
        probs=torch.softmax(logits,dim=-1)
        next_id=torch.multinomial(probs,1)
        ids=torch.cat([ids,next_id],dim=1)
    return ids
