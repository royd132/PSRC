"""Frozen Chronos-2 quantile inference, readout, and physical-unit calibration."""
from __future__ import annotations

import hashlib

import time

from pathlib import Path

import numpy as np

from formal.pc_fra import clip_prior, split_daylight_mask

CHRONOS2_QUANTILES: tuple[float, ...] = (
    0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
    0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99,
)

RECIPE_V4 = "v4"

BACKEND = "chronos-2"

MODEL_NAME = "amazon/chronos-2"

def model_revision(model_name: str) -> str | None:
    from huggingface_hub.constants import HF_HUB_CACHE

    ref = (
        Path(HF_HUB_CACHE)
        / f"models--{model_name.replace('/', '--')}"
        / "refs"
        / "main"
    )
    try:
        return ref.read_text(encoding="utf-8").strip()
    except OSError:
        return None

READOUT_Q50 = "q50"

READOUT_TRIMMED = "trimmed_mean_q20_q80"

READOUT_FULL_MEAN = "mean_q_full"

TRIM_LOW, TRIM_HIGH = 0.20, 0.80

SINGLE_Q_GRID: tuple[float, ...] = tuple(round(0.1 * k, 1) for k in range(1, 10))

SINGLE_Q_READOUTS: tuple[str, ...] = tuple(f"q{q:.2f}" for q in SINGLE_Q_GRID)

HORIZON_SEGMENTS: tuple[tuple[int, int], ...] = ((0, 4), (4, 8), (8, 12), (12, 16))

INPUT_NORMS: tuple[str, ...] = ("raw", "capacity")

def prior_cache_path_v4(
    dataset_path: str | Path,
    horizon: int,
    context_rows: int,
    input_norm: str,
) -> Path:
    if input_norm not in INPUT_NORMS:
        raise ValueError(f"input_norm must be one of {INPUT_NORMS}")
    digest = hashlib.sha256(Path(dataset_path).read_bytes()).hexdigest()[:8]
    return (
        Path("data/fm_prior_cache")
        / f"{digest}__{BACKEND}__h{horizon}__ctx{int(context_rows)}"
        f"__{input_norm}__prior_{RECIPE_V4}.npz"
    )

def valid_starts(starts: np.ndarray, seq_len: int, context_rows: int) -> np.ndarray:
    """Windows whose full Chronos context lies inside the dataset rows."""
    starts = np.asarray(starts, dtype=np.int64)
    return starts[starts + int(seq_len) - int(context_rows) >= 0]

def build_chronos2_grid_cache(
    bundle,
    starts: np.ndarray,
    horizon: int,
    context_rows: int,
    input_norm: str,
    *,
    seq_len: int = 96,
    batch_size: int = 128,
    force: bool = False,
    out_path: str | Path | None = None,
) -> Path:
    """Materialise one v4 cache: native 21-quantile grid per window.

    Context ends at the same forecast origin as PSRC's L96 history
    (``power[s+96-L_C : s+96]``). The capacity contract divides the input
    by the deterministic training capacity only; Chronos' internal
    scaling is untouched and outputs are multiplied back exactly.

    ``out_path`` writes a sidecar cache (same format) instead of the
    registered digest-keyed location; used for one-off split caches so
    development caches are never overwritten.
    """
    import torch
    from chronos import Chronos2Pipeline

    if input_norm not in INPUT_NORMS:
        raise ValueError(f"input_norm must be one of {INPUT_NORMS}")
    context_rows = int(context_rows)
    horizon = int(horizon)
    path = prior_cache_path_v4(bundle.path, horizon, context_rows, input_norm)
    if out_path is not None:
        path = Path(out_path)
    if path.is_file() and not force:
        return path

    requested = np.asarray(starts, dtype=np.int64)
    kept = valid_starts(requested, seq_len, context_rows)
    dropped = int(len(requested) - len(kept))
    power = np.nan_to_num(np.asarray(bundle.raw[:, -1]), nan=0.0).astype(np.float32)
    power = np.clip(power, 0.0, None)
    if input_norm == "capacity":
        scale = float(bundle.capacity)
        feed_power = power / scale
    else:
        scale = 1.0
        feed_power = power

    def window(start: int) -> np.ndarray:
        lo = int(start) + seq_len - context_rows
        return feed_power[lo : int(start) + seq_len]

    contexts = torch.from_numpy(np.stack([window(s) for s in kept])).unsqueeze(1)
    total = contexts.shape[0]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = Chronos2Pipeline.from_pretrained(MODEL_NAME, device_map=device)
    model_quantiles = [float(q) for q in pipeline.quantiles]
    if tuple(model_quantiles) != CHRONOS2_QUANTILES:
        raise ValueError(
            f"unexpected Chronos-2 quantile grid: {model_quantiles}"
        )

    grid = np.empty((total, len(CHRONOS2_QUANTILES), horizon), dtype=np.float16)
    started = time.perf_counter()
    for offset in range(0, total, batch_size):
        chunk = contexts[offset : offset + batch_size]
        with torch.inference_mode():
            outputs = pipeline.predict(
                chunk,
                prediction_length=horizon,
                batch_size=int(chunk.shape[0]),
            )
        block = torch.stack([item[0] for item in outputs], dim=0)
        block = block.detach().float().cpu().numpy().astype(np.float32)
        # Independent heads can cross near zero: enforce a monotone grid
        # along the quantile axis before any readout.
        block.sort(axis=1)
        if input_norm == "capacity":
            # Exact inverse of the deterministic input division, then clip.
            block *= scale
        block = np.maximum(block, 0.0)
        width = block.shape[0]
        grid[offset : offset + width] = block.astype(np.float16)
        done = offset + width
        if (offset // batch_size) % 20 == 0 or done == total:
            elapsed = time.perf_counter() - started
            rate = done / max(elapsed, 1e-9)
            eta = (total - done) / max(rate, 1e-9)
            print(
                f"[chronos2:v4] ctx={context_rows} norm={input_norm} "
                f"{done}/{total} ({done/total:.0%}) {rate:.0f} win/s "
                f"eta {eta/60:.1f} min",
                flush=True,
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        starts=kept,
        q_grid=grid,
        taus=np.array(CHRONOS2_QUANTILES, dtype=np.float64),
        dim=np.int64(horizon),
        kind=np.array("chronos2_quantile_grid"),
        backend=np.array(BACKEND),
        recipe=np.array(RECIPE_V4),
        model=np.array(MODEL_NAME),
        model_revision=np.array(model_revision(MODEL_NAME) or ""),
        dataset_sha256=np.array(
            hashlib.sha256(Path(bundle.path).read_bytes()).hexdigest()
        ),
        input_contract=np.array("target_pv_history_only"),
        input_norm=np.array(input_norm),
        capacity_scale=np.float64(scale),
        context_rows=np.int64(context_rows),
        seq_rows=np.int64(seq_len),
        n_requested_windows=np.int64(len(requested)),
        n_short_context_dropped=np.int64(dropped),
        num_samples=np.int64(0),
        sampling_seed=np.int64(-1),
        clip_negative_to_zero=np.array(True),
        quantile_ordering=np.array("sort21_monotone"),
    )
    return path

def load_grid_cache(path: str | Path):
    """Read a v4 cache; returns (starts, q_grid[N,Q,H] float32, metadata)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"missing v4 chronos grid cache: {path}")
    with np.load(path) as archive:
        if str(archive["recipe"]) != RECIPE_V4:
            raise ValueError(f"{path} is not a v4 cache")
        taus = np.asarray(archive["taus"], dtype=np.float64)
        if tuple(float(t) for t in taus) != CHRONOS2_QUANTILES:
            raise ValueError(f"{path} quantile grid mismatch")
        starts = np.asarray(archive["starts"], dtype=np.int64)
        q_grid = np.asarray(archive["q_grid"], dtype=np.float32)
        meta = {
            "cache_path": str(path),
            "context_rows": int(archive["context_rows"]),
            "input_norm": str(archive["input_norm"]),
            "capacity_scale": float(archive["capacity_scale"]),
            "model_revision": str(archive["model_revision"]),
            "quantile_ordering": str(archive["quantile_ordering"]),
            "n_windows": int(len(starts)),
            "n_short_context_dropped": int(archive["n_short_context_dropped"]),
        }
    return starts, q_grid, meta

def _trapz_mean(values: np.ndarray, taus: np.ndarray) -> np.ndarray:
    """Trapezoid integral of Q(tau) divided by the covered tau width."""
    trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    return trapz(values, taus, axis=1) / float(taus[-1] - taus[0])

def readout(q_grid: np.ndarray, taus: np.ndarray, name: str) -> np.ndarray:
    """Return one physical-unit prior trajectory [N, H]."""
    if name == READOUT_Q50:
        idx = tuple(float(t) for t in taus).index(0.50)
        return q_grid[:, idx, :].copy()
    if name == READOUT_FULL_MEAN:
        # Mean over the full native grid [0.01, 0.99] (width normalised).
        return _trapz_mean(q_grid, taus)
    if name == READOUT_TRIMMED:
        keep = (taus >= TRIM_LOW - 1e-12) & (taus <= TRIM_HIGH + 1e-12)
        sub_q = q_grid[:, keep, :]
        sub_t = taus[keep]
        return _trapz_mean(sub_q, sub_t)
    # Any "q0.35"-style name resolves to that native-grid quantile; the grid
    # membership check inside readout_quantile rejects off-grid levels.
    if name.startswith("q"):
        return readout_quantile(q_grid, taus, float(name[1:]))
    raise ValueError(f"unknown readout {name!r}")

def readout_quantile(q_grid: np.ndarray, taus: np.ndarray, q: float) -> np.ndarray:
    levels = [float(t) for t in taus]
    if float(q) not in levels:
        raise ValueError(f"quantile {q} is not on the native grid")
    return q_grid[:, levels.index(float(q)), :].copy()

def skew_readout_name(beta: float) -> str:
    return f"skew_b{beta:+.2f}"

def skew_readout(q_grid: np.ndarray, taus: np.ndarray, beta: float) -> np.ndarray:
    """Distribution-shape trajectory  P* = q50 + beta*(q90 + q10 - 2*q50).

    ``beta`` re-weights the asymmetry of Chronos' predictive distribution
    without learning any parameter from the target series; beta=0 returns
    the native median exactly. Output stays a single [N, H] physical-unit
    trajectory, so the downstream adapter contract is unchanged.
    """
    q10 = readout_quantile(q_grid, taus, 0.10)
    q50 = readout_quantile(q_grid, taus, 0.50)
    q90 = readout_quantile(q_grid, taus, 0.90)
    return (q50 + float(beta) * (q90 + q10 - 2.0 * q50)).astype(np.float32)

def scale_disagreement(
    prior_phys: np.ndarray, psrc_phys: np.ndarray, gamma: float
) -> np.ndarray:
    """P' = y_psrc + gamma*(P* - y_psrc), one scalar strength (L96 only)."""
    return (psrc_phys + float(gamma) * (prior_phys - psrc_phys)).astype(np.float32)

def scale_disagreement_segments(
    prior_phys: np.ndarray,
    psrc_phys: np.ndarray,
    gamma_groups: tuple[float, ...],
) -> np.ndarray:
    """Same calibration with one gamma per fixed HORIZON_SEGMENTS group."""
    if len(gamma_groups) != len(HORIZON_SEGMENTS):
        raise ValueError("one gamma per horizon segment required")
    out = np.array(psrc_phys, dtype=np.float32, copy=True)
    for gamma, (lo, hi) in zip(gamma_groups, HORIZON_SEGMENTS):
        out[:, lo:hi] = psrc_phys[:, lo:hi] + float(gamma) * (
            prior_phys[:, lo:hi] - psrc_phys[:, lo:hi]
        )
    return out

def contract_prior(
    prior_phys: np.ndarray,
    bundle,
    starts: np.ndarray,
    seq_len: int,
    horizon: int,
) -> np.ndarray:
    """Apply the registered PV contract: horizon-wise mask and clipping."""
    mask = split_daylight_mask(bundle, np.asarray(starts), seq_len, horizon)
    out = np.empty_like(prior_phys, dtype=np.float32)
    for i in range(len(starts)):
        out[i] = clip_prior(prior_phys[i], mask[i].astype(np.float32), bundle.capacity)
    return out
