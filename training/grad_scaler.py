from __future__ import annotations

class GradScalerAdapter:
    """Small adapter around torch.amp.GradScaler with safe CPU fallback."""
    def __init__(self, enabled: bool = True):
        import torch
        self.enabled = bool(enabled and torch.cuda.is_available())
        self._scaler = torch.amp.GradScaler('cuda', enabled=self.enabled)

    def scale(self, loss):
        return self._scaler.scale(loss)

    def step(self, optimizer):
        self._scaler.step(optimizer)

    def update(self):
        self._scaler.update()

    def unscale_(self, optimizer):
        self._scaler.unscale_(optimizer)
