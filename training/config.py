from dataclasses import dataclass, asdict
import json
from pathlib import Path

@dataclass
class TrainConfig:
    seed:int=42
    batch_size:int=2
    grad_accum:int=8
    steps:int=1000
    lr:float=3e-4
    min_lr:float=3e-5
    warmup_steps:int=100
    weight_decay:float=0.1
    grad_clip:float=1.0
    eval_every:int=100
    save_every:int=100
    out_dir:str='checkpoints'
    amp:bool=True

    def save(self,path):
        p=Path(path); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(asdict(self),indent=2))
