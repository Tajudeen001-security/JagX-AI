from __future__ import annotations
from pathlib import Path
import torch
from security.artifact_integrity import sha256_file


def save_checkpoint(model, optimizer, step: int, path: str, metadata: dict | None = None) -> dict:
    if step < 0:
        raise ValueError('step must be non-negative')
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + '.tmp')
    payload = {
        'step': step,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'metadata': metadata or {},
    }
    torch.save(payload, tmp)
    tmp.replace(target)
    return {'path': str(target), 'sha256': sha256_file(str(target)), 'step': step}


def load_checkpoint(path: str, device='cpu') -> dict:
    return torch.load(path, map_location=device, weights_only=True)
