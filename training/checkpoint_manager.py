from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json

@dataclass(frozen=True)
class CheckpointRecord:
    step:int
    path:str
    eval_score:float
    loss:float

class CheckpointManager:
    def __init__(self,root:str='checkpoints'):
        self.root=Path(root); self.root.mkdir(parents=True,exist_ok=True)
        self.manifest=self.root/'manifest.json'

    def record(self,step:int,path:str,eval_score:float,loss:float):
        if step<0: raise ValueError('step cannot be negative')
        if not all(map(lambda x: isinstance(x,(int,float)),(eval_score,loss))): raise TypeError('scores must be numeric')
        records=self.records(); records.append(CheckpointRecord(step,str(path),float(eval_score),float(loss)))
        self.manifest.write_text(json.dumps([r.__dict__ for r in records],indent=2),encoding='utf-8')

    def records(self):
        if not self.manifest.exists(): return []
        return [CheckpointRecord(**x) for x in json.loads(self.manifest.read_text(encoding='utf-8'))]

    def best(self):
        records=self.records(); return max(records,key=lambda r:r.eval_score) if records else None
