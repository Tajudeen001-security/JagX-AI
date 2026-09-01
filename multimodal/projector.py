from __future__ import annotations
import torch
from torch import nn

class VisionProjector(nn.Module):
    """Maps frozen/external vision features into JagX token space."""
    def __init__(self, vision_dim:int, model_dim:int, hidden_dim:int|None=None):
        super().__init__(); h=hidden_dim or model_dim*4
        self.net=nn.Sequential(nn.LayerNorm(vision_dim),nn.Linear(vision_dim,h),nn.GELU(),nn.Linear(h,model_dim))
    def forward(self, features:torch.Tensor)->torch.Tensor:
        if features.ndim!=3: raise ValueError('features must be [B,N,D]')
        return self.net(features)

class AudioProjector(VisionProjector):
    """Projects frame-level audio features into the same token space."""
    pass
