from __future__ import annotations
from dataclasses import dataclass
from collections import deque
import torch

@dataclass
class GenerationJob:
    input_ids: torch.Tensor
    max_new_tokens: int

class ContinuousBatcher:
    """Small dependency-free request batcher for local inference servers."""
    def __init__(self,max_batch_size:int=8):
        if max_batch_size<1: raise ValueError('max_batch_size must be positive')
        self.max_batch_size=max_batch_size; self.queue=deque()
    def submit(self,job:GenerationJob): self.queue.append(job)
    def pop_batch(self):
        jobs=[]
        while self.queue and len(jobs)<self.max_batch_size: jobs.append(self.queue.popleft())
        return jobs
    @staticmethod
    def pad(jobs,pad_id:int=0):
        if not jobs: return None
        n=max(j.input_ids.numel() for j in jobs); out=torch.full((len(jobs),n),pad_id,dtype=jobs[0].input_ids.dtype,device=jobs[0].input_ids.device)
        for i,j in enumerate(jobs): out[i,-j.input_ids.numel():]=j.input_ids
        return out
