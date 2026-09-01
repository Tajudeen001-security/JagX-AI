from __future__ import annotations
from pathlib import Path
import torch

class SafeModelLoader:
    """Loads tensor-only JagX checkpoints and refuses unsafe pickle artifacts."""
    def __init__(self,device=None): self.device=device or ('cuda' if torch.cuda.is_available() else 'cpu')
    def load_state_dict(self,path:str):
        p=Path(path)
        if p.suffix.lower() not in {'.safetensors','.pt','.pth'}: raise ValueError('unsupported model artifact')
        state=torch.load(p,map_location=self.device,weights_only=True)
        if not isinstance(state,dict): raise ValueError('checkpoint must contain a state dictionary')
        return state
