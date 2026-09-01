from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json

@dataclass
class Experiment:
    name:str
    model_config:str
    dataset_manifest:str
    output_dir:str
    seed:int=42

class ExperimentOrchestrator:
    """Creates reproducible experiment manifests; execution is delegated to the trainer."""
    def __init__(self,root='experiments'): self.root=Path(root)
    def create(self,exp:Experiment):
        p=self.root/exp.name; p.mkdir(parents=True,exist_ok=True)
        (p/'experiment.json').write_text(json.dumps(exp.__dict__,indent=2),encoding='utf-8')
        return p
