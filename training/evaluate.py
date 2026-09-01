from __future__ import annotations
import math
import torch

def evaluate(model,batches,device=None,max_batches=50):
    device=torch.device(device or next(model.parameters()).device)
    model.eval(); total=0.0; count=0
    with torch.no_grad():
        for x,y in batches:
            x,y=x.to(device),y.to(device)
            _,loss=model(x,y); total+=float(loss); count+=1
            if count>=max_batches: break
    model.train()
    mean=total/max(1,count)
    return {'loss':mean,'perplexity':math.exp(min(20.0,mean)),'batches':count}
