"""Horizon-wise physical-language guidance inside the late Chronos adapter."""
import math

import torch
from torch import nn
from torch.nn import functional as F


class PhysicsGuidedChronos(nn.Module):
    """Frozen forecast trajectories query six calibrated physical-language states."""

    def __init__(self, epsilon, statistics, *, gate=.5, temperature=.2, dropout=0.):
        super().__init__()
        if epsilon <= 0 or temperature <= 0 or not 0 < gate < 1:
            raise ValueError("Positive correction bound/temperature and interior gate required")
        self.epsilon, self.temperature = float(epsilon), float(temperature)
        for name, value in statistics.items():
            self.register_buffer(name, torch.as_tensor(value, dtype=torch.float32))
        self.physical = nn.Linear(32, 32)
        self.language = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 32))
        self.category = nn.Parameter(torch.randn(1, 6, 32) * .02)
        self.horizon = nn.Parameter(torch.randn(1, 16, 32) * .02)
        self.gates = nn.Parameter(torch.full((6, 1), math.log(gate/(1-gate))))
        self.query_global = nn.Linear(48, 32)
        self.query_local = nn.Linear(4, 32)
        self.hidden = nn.Sequential(nn.Linear(68, 64), nn.GELU(), nn.Dropout(dropout))
        self.output = nn.Linear(64, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, old, prior, disagreement, physical, language, available):
        if old.ndim != 2 or old.shape[1] != 16:
            raise ValueError("L96/H16 study requires [B,16] forecasts")
        if not available.bool().any(1).all():
            raise ValueError("At least one valid physical-language category is required")
        p = self.physical(((physical-self.physical_mu)/self.physical_sd).clamp(-5, 5))
        t = self.language(((language-self.language_mu)/self.language_sd).clamp(-5, 5))
        states = p + self.gates.sigmoid() * t + self.category
        trajectories = torch.cat((old, prior, disagreement), -1)
        trajectories = ((trajectories-self.trajectory_mu)/self.trajectory_sd).clamp(-5, 5)
        local = trajectories.reshape(len(old), 3, 16).transpose(1, 2)
        ramps = F.pad(local[:, 1:, 1]-local[:, :-1, 1], (1, 0))
        local = torch.cat((local, ramps.unsqueeze(-1)), -1)
        query = self.query_global(trajectories).unsqueeze(1) + self.query_local(local) + self.horizon
        logits = F.normalize(query, dim=-1) @ F.normalize(states, dim=-1).transpose(1, 2)
        logits = logits / self.temperature
        logits = logits.masked_fill(~available.bool().unsqueeze(1), -torch.inf)
        attention = logits.softmax(-1)
        retrieved = attention @ F.layer_norm(states, (32,))
        hidden = self.hidden(torch.cat((query, retrieved, local), -1))
        delta = self.epsilon * self.output(hidden).squeeze(-1).tanh()
        return old + delta, delta, p, t, attention
