"""Compact trainable cross-attention over frozen Qwen language states."""
from torch import nn

from models.physical_foundation import PhysicsGuidedChronos


class QwenGuidedChronos(PhysicsGuidedChronos):
    def __init__(self, epsilon, statistics, **kwargs):
        super().__init__(epsilon, statistics, **kwargs)
        self.language = nn.Sequential(nn.LayerNorm(64), nn.Linear(64, 32))
