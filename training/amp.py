from __future__ import annotations


def autocast_context(enabled: bool = True, dtype_name: str = 'bf16'):
    """Return PyTorch autocast context without importing torch until training runs."""
    import torch
    if not enabled:
        return torch.autocast(device_type='cpu', enabled=False)
    if dtype_name == 'bf16':
        dtype = torch.bfloat16
    elif dtype_name == 'fp16':
        dtype = torch.float16
    else:
        raise ValueError('dtype_name must be bf16 or fp16')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return torch.autocast(device_type=device, dtype=dtype, enabled=True)
