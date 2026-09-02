from __future__ import annotations


def autocast_context(
    enabled: bool = True,
    dtype_name: str = "bf16",
    device_type: str | None = None,
):
    """Return a device-correct PyTorch autocast context."""
    import torch

    device = device_type or ("cuda" if torch.cuda.is_available() else "cpu")
    if device not in {"cpu", "cuda", "mps"}:
        raise ValueError("device_type must be cpu, cuda, mps, or None")
    if not enabled:
        return torch.autocast(device_type=device, enabled=False)
    if dtype_name == "bf16":
        dtype = torch.bfloat16
    elif dtype_name == "fp16":
        dtype = torch.float16
    else:
        raise ValueError("dtype_name must be bf16 or fp16")
    return torch.autocast(device_type=device, dtype=dtype, enabled=True)
