"""Optional late correction; never changes the frozen PSRC/Chronos path."""
import torch
from torch import nn
from torch.nn import functional as F


def state_alignment(physical, language, codes, available, temperature=.1):
    """Same-category multi-positive alignment; equal states are NOT negatives.

    Codes are training-thresholded historical descriptors, not future labels.
    A batch/category with no distinct states contributes no alignment loss.
    """
    terms = []
    for k in range(physical.shape[1]):
        keep = available[:, k].bool()
        labels = codes[keep, k]
        if len(labels) < 2 or labels.unique().numel() < 2:
            continue
        p = F.normalize(physical[keep, k], dim=-1)
        t = F.normalize(language[keep, k], dim=-1)
        scores = p @ t.T / temperature
        positive = labels[:, None] == labels[None, :]
        for logits, mask in ((scores, positive), (scores.T, positive.T)):
            terms.append((logits.logsumexp(1) -
                          logits.masked_fill(~mask, -torch.inf).logsumexp(1)).mean())
    return torch.stack(terms).mean() if terms else physical.sum() * 0.


class PhysicalLanguageCorrection(nn.Module):
    """Six category-preserving pairs -> bounded residual on the OLD forecast.

    All controls use this exact trainable architecture. Only the frozen
    representation and the auxiliary loss change. The final output layer,
    not all upstream gates, is zero-initialized so learning can start.
    """
    def __init__(self, epsilon, horizon=16, physical_dim=32, text_dim=768):
        super().__init__()
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        self.epsilon = float(epsilon)
        self.horizon = horizon
        self.physical = nn.Linear(physical_dim, 32)
        self.language = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, 32))
        self.gates = nn.Parameter(torch.full((6, 1), -2.))
        self.hidden = nn.Sequential(nn.Linear(6 * 32 + 3 * horizon, 64), nn.GELU())
        self.output = nn.Linear(64, horizon)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, old, prior, disagreement, physical, language, available):
        if old.ndim != 2 or old.shape[1] != self.horizon:
            raise ValueError("old forecast must be [B,H]")
        if physical.shape[:2] != (len(old), 6) or language.shape[:2] != (len(old), 6):
            raise ValueError("physical/language inputs must retain six categories")
        p, t = self.physical(physical), self.language(language)
        fused = (p + self.gates.sigmoid() * t) * available.unsqueeze(-1)
        features = torch.cat((old, prior, disagreement, fused.flatten(1)), dim=-1)
        delta = self.epsilon * self.output(self.hidden(features)).tanh()
        return old + delta, delta, p, t
