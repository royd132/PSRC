"""Physics-aware semantic tokens for residual calibration (PSRC).

The semantic probe showed that GPT-2 sentence embeddings do not carry
residual-predictive signal for this task, while six structured
physical-state tokens do.  This module is the single source of truth for
those tokens: the probe, the Optuna-validated runner, and the model all
share :func:`compute_physical_tokens`, so a probe conclusion can never
drift from what the production model consumes.

Tokens (all strictly causal, computed from the input window only):

    0 ramp_state                  recent ramp intensity + signed net changes
    1 cloud_transition            clear-sky index statistics in daylight
    2 irradiance_power_coupling   full / recent / lagged correlation
    3 thermal                     temperature level, drift, power coupling
    4 diurnal_phase               solar geometry at forecast issue time
    5 multi_scale_volatility      detrended power std at 1h/3h/6h scales
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

IRRADIANCE_CANDIDATES = (
    "Radiation_Global_Tilted", "Global_Horizontal_Radiation",
    "lmd_totalirrad", "GHI",
)
CLEAR_SKY_CANDIDATES = ("ClearSkyGHIProxy", "ClearSkyProxy", "SolarGeometryProxy")
TEMPERATURE_CANDIDATES = (
    "Weather_Temperature_Celsius", "lmd_temperature",
    "Ambient_Temperature", "Module_Temperature",
)
DAY_SIN_CANDIDATES = ("SolarDaySin", "ClockSin")
DAY_COS_CANDIDATES = ("SolarDayCos", "ClockCos")
ZENITH_CANDIDATES = ("SolarZenithCos", "SolarElevationMask")

N_TOKENS = 6
TOKEN_WIDTH = 6

_ROLE_CANDIDATES = {
    "irradiance": IRRADIANCE_CANDIDATES,
    "clear_sky": CLEAR_SKY_CANDIDATES,
    "temperature": TEMPERATURE_CANDIDATES,
    "day_sin": DAY_SIN_CANDIDATES,
    "day_cos": DAY_COS_CANDIDATES,
    "zenith": ZENITH_CANDIDATES,
}


def resolve_physical_roles(feature_names) -> dict[str, int | None]:
    """Map each physical role to its column index (None when absent)."""
    names = list(feature_names)
    return {
        role: next(
            (names.index(candidate) for candidate in candidates
             if candidate in names),
            None,
        )
        for role, candidates in _ROLE_CANDIDATES.items()
    }


def _corr(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Row-wise Pearson correlation of [batch, length] matrices."""
    a = a - a.mean(dim=1, keepdim=True)
    b = b - b.mean(dim=1, keepdim=True)
    denom = (a * a).sum(dim=1).sqrt() * (b * b).sum(dim=1).sqrt()
    return torch.where(
        denom > 1e-12, (a * b).sum(dim=1) / denom.clamp_min(1e-12),
        torch.zeros_like(denom),
    )


def _moving_mean(power: torch.Tensor, window: int) -> torch.Tensor:
    """Zero-padded 'same' moving average, matching numpy convolve semantics."""
    length = power.shape[1]
    kernel = power.new_ones((1, 1, window)) / float(window)
    padded = F.conv1d(power.unsqueeze(1), kernel, padding=window // 2)
    return padded[:, 0, :length]


@torch.no_grad()
def compute_physical_tokens(
    x_std: torch.Tensor,
    roles: dict[str, int | None],
    feature_mu,
    feature_sd,
    target_mu: float,
    target_sd: float,
) -> torch.Tensor:
    """Six physical-state tokens per window from standardized features.

    Parameters
    ----------
    x_std : [batch, length, channels] standardized input window; the target
        is always the last channel (data_windows appends it last).
    roles : output of :func:`resolve_physical_roles` for these channels.
    feature_mu / feature_sd : per-channel destandardization statistics.
    """
    batch, length, channels = x_std.shape
    mu = torch.as_tensor(feature_mu, dtype=x_std.dtype, device=x_std.device)
    sd = torch.as_tensor(feature_sd, dtype=x_std.dtype, device=x_std.device)

    def physical(idx: int) -> torch.Tensor:
        return x_std[:, :, idx] * sd[idx] + mu[idx]

    power = x_std[:, :, channels - 1] * float(target_sd) + float(target_mu)
    tokens = x_std.new_zeros((batch, N_TOKENS, TOKEN_WIDTH))

    ramp = (power[:, 1:] - power[:, :-1]).abs()
    tokens[:, 0, :5] = torch.stack([
        ramp[:, -4:].mean(dim=1), ramp[:, -12:].mean(dim=1),
        ramp[:, -24:].mean(dim=1),
        power[:, -1] - power[:, -4], power[:, -1] - power[:, -12],
    ], dim=1)

    sky = roles.get("clear_sky")
    if sky is not None:
        clear = physical(sky)
        threshold = torch.maximum(
            clear.max(dim=1, keepdim=True).values * 0.05,
            clear.new_full((batch, 1), 1e-3),
        )
        daylight = clear > threshold
        count = daylight.sum(dim=1)
        csi = torch.clamp(power / clear.clamp_min(1e-3), 0.0, 2.0)
        enough = (count >= 8) & daylight.any(dim=1)
        weight = daylight.to(x_std.dtype)
        denom = count.clamp_min(1).unsqueeze(1)
        csi_mean = (csi * weight).sum(dim=1, keepdim=True) / denom
        csi_std = (
            (csi * csi * weight).sum(dim=1, keepdim=True) / denom - csi_mean ** 2
        ).clamp_min(0.0).sqrt()
        # cloud transitions only between adjacent daylight steps
        neighbor = daylight[:, 1:] & daylight[:, :-1]
        transitions = (csi[:, 1:] - csi[:, :-1]).abs()
        n_trans = neighbor.sum(dim=1).clamp_min(1)
        trans_mean = (transitions * neighbor).sum(dim=1) / n_trans
        trans_frac = (
            ((transitions > 0.15) & neighbor).sum(dim=1).to(x_std.dtype)
            / n_trans
        )
        # last daylight clear-sky index vs its first-quarter mean
        last_idx = length - 1 - daylight.to(torch.long).flip(1).argmax(dim=1)
        last_csi = csi.gather(1, last_idx.unsqueeze(1)).squeeze(1)
        rank = daylight.to(torch.long).cumsum(dim=1) - 1
        first_quarter = daylight & (
            rank < (count // 4).clamp_min(1).unsqueeze(1)
        )
        fq_denom = first_quarter.sum(dim=1).clamp_min(1)
        fq_mean = (csi * first_quarter).sum(dim=1) / fq_denom
        tokens[:, 1, :5] = torch.stack([
            csi_mean.squeeze(1), csi_std.squeeze(1), trans_mean, trans_frac,
            last_csi - fq_mean,
        ], dim=1) * enough.to(x_std.dtype).unsqueeze(1)

    irradiance = roles.get("irradiance")
    if irradiance is not None:
        irr = physical(irradiance)
        tokens[:, 2, :3] = torch.stack([
            _corr(irr, power),
            _corr(irr[:, -24:], power[:, -24:]),
            _corr(irr[:, :-1], power[:, 1:]),
        ], dim=1)

    temperature = roles.get("temperature")
    if temperature is not None:
        temp = physical(temperature)
        tokens[:, 3, :3] = torch.stack([
            temp.mean(dim=1), temp[:, -1] - temp.mean(dim=1),
            _corr(temp, power),
        ], dim=1)

    phase = [
        x_std[:, -1, roles[role]]
        for role in ("day_sin", "day_cos", "zenith")
        if roles.get(role) is not None
    ]
    if phase:
        tokens[:, 4, :len(phase)] = torch.stack(phase, dim=1)

    detrended = power - _moving_mean(power, 12)
    tokens[:, 5, :3] = torch.stack([
        detrended[:, -24:].std(dim=1, correction=0),
        detrended[:, -48:].std(dim=1, correction=0),
        detrended.std(dim=1, correction=0),
    ], dim=1)
    return tokens


@torch.no_grad()
def compute_generic_tokens(x_std: torch.Tensor) -> torch.Tensor:
    """Six chronological blocks, each with six channel-averaged statistics."""
    tokens = []
    for block in torch.tensor_split(x_std, 6, dim=1):
        time = torch.linspace(-1, 1, block.shape[1], device=block.device)
        slope = (block * time[None, :, None]).mean(1) / time.square().mean()
        tokens.append(torch.stack([
            block.mean(1).mean(-1), block.std(1, correction=0).mean(-1),
            block.amin(1).mean(-1), block.amax(1).mean(-1),
            block[:, -1].mean(-1), slope.mean(-1),
        ], dim=-1))
    return torch.stack(tokens, dim=1)


def physical_token_statistics(
    x_std, starts, seq_len: int, roles, feature_mu, feature_sd,
    target_mu: float, target_sd: float, chunk_size: int = 512,
    token_statistics: str = "physical",
) -> tuple[np.ndarray, np.ndarray]:
    """Training-window mean and rms scale of every token channel.

    Mirrors the probe's ``_normalize_token_bank`` so the model consumes
    tokens on exactly the same scale the validated probe used.
    """
    x = torch.as_tensor(np.asarray(x_std), dtype=torch.float32)
    starts = np.asarray(starts, dtype=np.int64)
    offset = torch.arange(seq_len)
    total = torch.zeros((N_TOKENS, TOKEN_WIDTH), dtype=torch.float64)
    total_sq = torch.zeros((N_TOKENS, TOKEN_WIDTH), dtype=torch.float64)
    count = 0
    for begin in range(0, len(starts), chunk_size):
        chunk = torch.as_tensor(
            starts[begin:begin + chunk_size], dtype=torch.long,
        )
        index = chunk[:, None] + offset[None, :]
        windows = x[index]
        tokens = compute_generic_tokens(windows) if token_statistics == "generic" else compute_physical_tokens(
            windows, roles, feature_mu, feature_sd, target_mu, target_sd,
        )
        total += tokens.double().sum(dim=0)
        total_sq += (tokens.double() ** 2).sum(dim=0)
        count += len(chunk)
    mean = (total / max(count, 1)).numpy()
    scale = np.sqrt((total_sq / max(count, 1)).numpy() - mean ** 2)
    return mean.astype(np.float32), np.maximum(scale, 1e-3).astype(np.float32)


class PhysicalSemanticEncoder(nn.Module):
    """Physical Semantic Token Encoder (PSRC component 1).

    The token computation is a fixed causal feature transform (no
    gradients); the learned projection maps each 6-wide physical state
    token into the ``d_token`` space consumed by the semantic residual
    calibration head.
    """

    def __init__(
        self,
        roles: dict[str, int | None],
        feature_mu,
        feature_sd,
        target_mu: float,
        target_sd: float,
        token_mean,
        token_scale,
        d_token: int = 32,
        dropout: float = 0.0,
        token_drop: int | None = None,
        token_mode: str = "real",
        token_statistics: str = "physical",
        regroup_seed: int | None = None,
    ) -> None:
        super().__init__()
        if token_mode not in ("real", "shuffle", "random"):
            raise ValueError(f"unknown token_mode {token_mode!r}")
        if token_drop is not None and not 0 <= int(token_drop) < N_TOKENS:
            raise ValueError("token_drop must index one of the six categories")
        self.roles = dict(roles)
        self.target_mu = float(target_mu)
        self.target_sd = float(target_sd)
        self.token_drop = None if token_drop is None else int(token_drop)
        self.token_mode = str(token_mode)
        if token_statistics not in ("physical", "generic"):
            raise ValueError("unknown token statistics")
        self.token_statistics = token_statistics
        self.regroup_seed = regroup_seed
        order = torch.arange(36)
        if regroup_seed is not None:
            order = torch.randperm(36, generator=torch.Generator().manual_seed(regroup_seed))
        self.register_buffer("regroup_order", order, persistent=False)
        # Supplementary negative controls (shuffle/random) use a private RNG
        # stream so the global dropout stream stays identical across arms.
        self._perm_counter = 0
        self.register_buffer(
            "feature_mu", torch.as_tensor(feature_mu, dtype=torch.float32),
        )
        self.register_buffer(
            "feature_sd", torch.as_tensor(feature_sd, dtype=torch.float32),
        )
        self.register_buffer(
            "token_mean",
            torch.as_tensor(token_mean, dtype=torch.float32).reshape(
                N_TOKENS, TOKEN_WIDTH,
            ),
        )
        self.register_buffer(
            "token_scale",
            torch.as_tensor(token_scale, dtype=torch.float32).reshape(
                N_TOKENS, TOKEN_WIDTH,
            ).clamp_min(1e-3),
        )
        self.d_token = int(d_token)
        self.projection = nn.Sequential(
            nn.Linear(TOKEN_WIDTH, d_token),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_token, d_token),
        )

    def standardized_tokens(self, x_enc: torch.Tensor) -> torch.Tensor:
        """Fixed causal token transform followed by training normalization."""
        raw = compute_generic_tokens(x_enc) if self.token_statistics == "generic" else compute_physical_tokens(
            x_enc, self.roles, self.feature_mu, self.feature_sd,
            self.target_mu, self.target_sd,
        )
        normalized = (raw - self.token_mean) / self.token_scale
        if self.regroup_seed is not None:
            normalized = normalized.flatten(1)[:, self.regroup_order].reshape(-1, 6, 6)
        return normalized

    def _permuted(self, projected: torch.Tensor) -> torch.Tensor:
        batch = projected.shape[0]
        generator = torch.Generator(device="cpu")
        generator.manual_seed(2026 + self._perm_counter)
        self._perm_counter += 1
        if self.token_mode == "shuffle":
            # Permute whole token sets across windows: the joint distribution
            # of physical states is preserved, only window correspondence is
            # destroyed.
            perm = torch.randperm(batch, generator=generator)
            return projected[perm.to(projected.device)]
        # "random": independently permute every projected coordinate across
        # the batch. Each coordinate's empirical marginal is preserved while
        # all within-window and cross-coordinate structure is destroyed.
        out = torch.empty_like(projected)
        for idx in range(projected.shape[1]):
            for col in range(projected.shape[2]):
                perm = torch.randperm(batch, generator=generator)
                out[:, idx, col] = projected[perm.to(projected.device), idx, col]
        return out

    def forward(self, x_enc: torch.Tensor) -> torch.Tensor:
        tokens = self.standardized_tokens(x_enc)
        if self.token_drop is not None:
            # Zero after standardization == impute this category with its
            # training-window mean; the projection sees an identical layout.
            tokens = tokens.clone()
            tokens[:, self.token_drop, :] = 0.0
        projected = self.projection(tokens)
        if self.token_mode != "real":
            projected = self._permuted(projected)
        return projected
