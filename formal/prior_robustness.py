"""Fixed-architecture follow-up: matched prior controls and train-only corruption.

Never promotes configurations using test. Reuses the frozen campaign engine;
only this study's training feature packs are modified, not model code.
"""
from __future__ import annotations

import csv
import hashlib
import json
from contextlib import contextmanager
from unittest.mock import patch

import numpy as np
import torch

from formal import evidence_study as ev
from formal import fm_campaign as fm
from formal.config import load_best_parameters, load_experiment_config
from formal.engine import _seed_everything
from layers.pc_fra import pack_slices

OUT = ev.ROOT / "outputs/prior_robustness_v2"
ARMS = ("clean", "noise", "state")
SEEDS = (4101, 4102, 4103, 4104, 4105)


def features(bundle, starts):
    return np.asarray([fm.window_scalar_features(bundle, int(s), 96, 16)
                       for s in starts], dtype=np.float64)


def restore_evaluation_flags(model, split):
    """Match historical execution, not just weights (GPU kernels can differ).

    train_one validation froze PSRC requires_grad; legacy test rebuild did not.
    Both paths evaluate under no_grad and NEVER update any parameter.
    """
    if split not in ("validation", "test"):
        raise ValueError(split)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(split == "test" or name.startswith("pc_fra_adapter."))


def matched_indices(starts, values, volatility_edges, seed):
    """Matched replacement (not marginal-preserving permutation), no targets.

    Match month, six-hour bucket, operating state, 10%-capacity level bin,
    and training-defined volatility tercile. Exclude overlapping L96+H16.
    Unmatched rows stay unchanged and are excluded from matched-only metrics.
    """
    starts = np.asarray(starts)
    groups = {}
    for i, v in enumerate(values):
        key = (int(v[5]), int(v[4]), int(v[3]), int(np.floor(v[0] / .1)),
               int(np.searchsorted(volatility_edges, v[1], side="right")))
        groups.setdefault(key, []).append(i)
    rng = np.random.default_rng(seed)
    donor = np.arange(len(starts))
    for members in groups.values():
        members = np.asarray(members)
        for i in members:
            valid = members[np.abs(starts[members] - starts[i]) >= 112]
            if len(valid):
                donor[i] = rng.choice(valid)
    return donor, donor != np.arange(len(starts))


def corrupt_prior(prior, values, capacity, arm, seed=2026):
    """Static 50% training augmentation; equal pre-projection noise energy.

    Generic noise and state-shaped errors share mask/sign/amplitude. Targets
    never enter this function. Shape differs; post-projection RMS is audited.
    """
    if arm not in ARMS or capacity <= 0 or not np.isfinite(prior).all():
        raise ValueError("Invalid corruption input")
    if arm == "clean":
        return prior.copy(), np.zeros(len(prior), dtype=bool)
    rng = np.random.default_rng(seed)
    selected = rng.random(len(prior)) < .5
    sign = rng.choice([-1., 1.], len(prior))
    amplitude = .03 * capacity * np.where(values[:, 3] == 0, .5, 1.)
    if arm == "noise":
        shape = rng.normal(size=prior.shape)
    else:
        shape = np.ones_like(prior)
        t = np.linspace(-1., 1., prior.shape[1])
        shape[values[:, 3] == 1] = t
        shape[values[:, 3] == 2] = 1. - t*t
    shape = shape / np.maximum(np.sqrt(np.mean(shape*shape, axis=1, keepdims=True)), 1e-8)
    return (prior + selected[:, None]*sign[:, None]*amplitude[:, None]*shape).astype(np.float32), selected


def freeze_protocol(config):
    # Refuse stale v1 inputs and require the previous campaign to be complete.
    ev.manifest(config)
    files = [ev.OUT / "protocol.json", config.best_parameters,
             ev.ROOT / "configs/formal_experiment.json",
             ev.ROOT / "formal/prior_robustness.py", ev.ROOT / "formal/pc_fra.py",
             ev.ROOT / "formal/chronos_prior.py"]
    for station in ev.STATIONS:
        for backend, arm in ev.ARMS:
            directory = ev.OUT / ev.tag(backend, arm) / station
            files += [directory / n for n in ("lock.json", "adapter_checkpoint.pt", "test.json")]
    value = dict(status="supplementary_after_test_exposure", stations=ev.STATIONS,
        diagnostic_backends=ev.BACKENDS, diagnostic_seeds=SEEDS,
        diagnostic_splits=["validation", "test"], match="month/bucket/state/level_0.1/train_sigma_tercile",
        exclude_overlap_steps=112, unmatched="identity, excluded from matched-only metrics",
        training_arms=ARMS, backend="chronos2", seed=2026, epochs=100,
        patience=101, checkpoint="minimum validation MSE; alpha validation-only",
        training_test_access=False, augmentation="static 50% train rows; RMS .03 capacity, half for low state",
        state_shapes="night_low/regular: bias; ramp: linear; peak: smooth hump",
        budget="same 100 epochs, batches, windows and optimiser schedule for all three arms",
        selection="report every arm, no automatic promotion or additional test evaluation",
        reconstruction_tolerance="1e-5 absolute standardized pointwise; 1e-6 physical RMSE/MAE/MBE/R2",
        execution="restore historical validation requires_grad freeze; preserve legacy test flags, both no_grad",
        known_limit="Frozen PSRC training residuals are in-sample, not rolling out-of-fold",
        sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    path = OUT / "protocol.json"
    value = json.loads(json.dumps(value))
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != value:
        raise ValueError("Follow-up protocol changed; use a new output version")
    ev.write_json(path, value)


def save_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def daily_errors(bundle, starts, pred, target):
    days = bundle.timestamps[np.asarray(starts) + 95].astype("datetime64[D]")
    unique = np.unique(days)
    return [dict(day=str(day), mse=float(np.mean((pred[days == day]-target[days == day])**2)),
                 mae=float(np.mean(np.abs(pred[days == day]-target[days == day])))) for day in unique]


def diagnose(config, records):
    rows = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for station in ev.STATIONS:
        bundle = fm._load_bundle(station, config)
        fv_train = features(bundle, bundle.train.starts)
        edges = np.quantile(fv_train[:, 1], [1/3, 2/3])
        ev.write_json(OUT / "diagnostics" / station / "state_audit.json", dict(
            volatility_edges_train=edges.tolist(), capacity=float(bundle.capacity),
            definitions="history-only: low<.2 capacity; ramp>.1; peak>.8; otherwise regular",
            counts_train=np.bincount(fv_train[:, 3].astype(int), minlength=4).tolist()))
        params = records[station]["params"]
        state = torch.load(fm._checkpoint(station), map_location="cpu", weights_only=True)["state_dict"]
        frozen, _ = fm._build_frozen(station, config, bundle, params, state, device)
        batch = int(params.get("batch", 128))
        sd, mu = float(bundle.feature_sd[-1]), float(bundle.feature_mu[-1])
        for split in ("validation", "test"):
            starts = getattr(bundle, split).starts
            fv = features(bundle, starts)
            a = fm._infer(frozen, bundle, starts, batch, device)[1]
            psrc, target = a["pred_std"], a["target_std"]
            pphys = psrc*sd+mu
            for backend in ev.BACKENDS:
                print(f"DIAGNOSE {station} {split} {backend}", flush=True)
                directory = ev.OUT / ev.tag(backend, "real") / station
                lock = json.loads((directory / "lock.json").read_text())
                model, _ = fm._rebuild_model(station, config, records, bundle, state, lock, device)
                model.pc_fra_adapter.load_state_dict(torch.load(directory / "adapter_checkpoint.pt",
                    map_location=device, weights_only=True)["state_dict"])
                restore_evaluation_flags(model, split)
                cs, grid, _ = fm.load_grid_cache(ev.cache_path(station, backend, config,
                    "dev" if split == "validation" else "test"))
                lookup = {int(s): i for i, s in enumerate(cs)}
                raw_map = fm._calibrated_lookup(ev.SETTINGS[station][0], ("scalar", 1.),
                    grid[[lookup[int(s)] for s in starts]], np.asarray(fm.CHRONOS2_QUANTILES),
                    bundle, starts, {"pred_phys": pphys})
                raw = np.stack([raw_map[int(s)] for s in starts])
                original_prediction = None
                for seed in (0,) + SEEDS:
                    out = OUT / "diagnostics" / station / split / backend / str(seed)
                    donor, matched = (np.arange(len(starts)), np.ones(len(starts), dtype=bool)) if seed == 0 else matched_indices(starts, fv, edges, seed)
                    prior = ev.controlled_prior(raw[donor], pphys, "real", 0)
                    # No extra clipping: identical contract to the v1 real arm.
                    with ev.prior_hooks(station, backend, "real", config):
                        pack = fm.build_pack(bundle, starts, 96, 16,
                            {int(s): prior[i] for i, s in enumerate(starts)},
                            {int(s): psrc[i] for i, s in enumerate(starts)}, starts)
                    pred = fm._evaluate(model, fm._loader(bundle, starts, batch, False, 2026,
                        fm_embeddings=pack), bundle, device, collect=True)[1]["pred_std"]
                    pred = psrc + lock["alpha"]*(pred-psrc)
                    reconstruction = None
                    if seed == 0:
                        original_prediction = pred
                        with np.load(directory / ("test_arrays.npz" if split == "test" else "validation_arrays.npz")) as old:
                            reference = old["pred_std"] if split == "test" else old["psrc_std"] + lock["alpha"]*(old["pred_b_std"]-old["psrc_std"])
                        # The old train_one inferred its first PSRC arrays before
                        # enabling deterministic kernels. Audit, do not demand
                        # bitwise equality across those GPU evaluation paths.
                        np.testing.assert_allclose(pred, reference, atol=1e-5, rtol=0)
                        new_metrics = fm._phys_metrics(psrc, pred, target, sd, mu, bundle.capacity)
                        old_metrics = fm._phys_metrics(psrc, reference, target, sd, mu, bundle.capacity)
                        np.testing.assert_allclose(list(new_metrics.values()), list(old_metrics.values()),
                                                   atol=1e-6, rtol=0)
                        reconstruction = dict(max_abs_std=float(np.max(np.abs(pred-reference))),
                                              actual=new_metrics, reference=old_metrics)
                    out.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(out / "arrays.npz", starts=starts, donor_starts=starts[donor],
                        matched=matched, pred_std=pred, target_std=target, target_sd=sd, target_mu=mu)
                    m = fm._phys_metrics(psrc, pred, target, sd, mu, bundle.capacity)
                    matched_metrics, matched_real = {}, {}
                    if matched.any():
                        args = (target[matched], sd, mu, bundle.capacity)
                        matched_metrics = fm._phys_metrics(psrc[matched], pred[matched], *args)
                        matched_real = fm._phys_metrics(psrc[matched], original_prediction[matched], *args)
                    ev.write_json(out / "report.json", dict(metrics=m, matched_metrics=matched_metrics,
                        matched_real=matched_real, coverage=float(matched.mean()), alpha=lock["alpha"],
                        reconstruction=reconstruction,
                        daily=daily_errors(bundle, starts, np.maximum(0, pred*sd+mu), target*sd+mu),
                        note="Fixed-checkpoint input intervention, not a retrained control or causal proof"))
                    rows.append(dict(station=station, split=split, backend=backend, seed=seed,
                        coverage=float(matched.mean()), **m,
                        matched_rmse=matched_metrics.get("rmse_physical"),
                        matched_real_rmse=matched_real.get("rmse_physical")))
                    save_csv(OUT / "matched_comparison.csv", rows)


@contextmanager
def training_hooks(station, arm, config):
    with ev.prior_hooks(station, "chronos2", "real", config):
        original = fm.build_pack
        def pack(bundle, starts, seq_len, horizon, median_lookup, psrc_lookup, partners):
            result = original(bundle, starts, seq_len, horizon, median_lookup, psrc_lookup, partners)
            if np.array_equal(starts, bundle.train.starts):
                sl = pack_slices(horizon)["c_bar"]
                old = result.embeddings[:, sl].copy()
                changed, selected = corrupt_prior(old, features(bundle, starts), bundle.capacity, arm)
                if arm != "clean":
                    projected = fm.contract_prior(changed, bundle, starts, seq_len, horizon)
                    changed[selected] = projected[selected]
                result.embeddings[:, sl] = changed
                ev.write_json(OUT / arm / station / "augmentation.json", dict(
                    selected_fraction=float(selected.mean()),
                    rms_physical=float(np.sqrt(np.mean((changed-old)**2))),
                    validation_augmented=False, disagreement="recomputed by existing model forward"))
            elif not np.array_equal(starts, bundle.validation.starts):
                raise ValueError("Training pack must be exactly train or validation")
            return result
        with patch.object(fm, "build_pack", pack), patch.object(fm, "PATIENCE", 101):
            yield


def train(config, records):
    rows = []
    for station in ev.STATIONS:
        cand, lr, wd, _, beta, energy, smooth = ev.SETTINGS[station]
        baseline_lock = json.loads((ev.OUT / "chronos2__real" / station / "lock.json").read_text())
        capacity = float(fm._load_bundle(station, config).capacity)
        for arm in ARMS:
            print(f"TRAIN {station} {arm} fixed100", flush=True)
            with training_hooks(station, arm, config):
                result, arrays = fm.train_one(station, config, records, cand, ("scalar", .5), arm, OUT,
                    lr=lr, weight_decay=wd, epsilon=baseline_lock["epsilon"], huber_beta_c=beta,
                    lambda_energy=energy, lambda_smooth=smooth)
            if result["epochs_ran"] != 100:
                raise ValueError("Equal-budget experiment did not run 100 epochs")
            alpha = fm.shrinkage_analysis(arrays)["alpha_star"]
            pred = arrays["psrc_std"] + alpha*(arrays["pred_b_std"]-arrays["psrc_std"])
            metrics = fm._phys_metrics(arrays["psrc_std"], pred, arrays["target_std"],
                float(arrays["target_sd"]), float(arrays["target_mu"]), capacity)
            ev.write_json(OUT / arm / station / "lock.json", dict(baseline_lock, alpha=alpha,
                source="validation_only", metrics=metrics, arm=arm))
            rows.append(dict(station=station, arm=arm, alpha=alpha,
                best_epoch=result["best_epoch"], epochs_ran=100, **metrics))
            save_csv(OUT / "validation_comparison.csv", rows)


def main(action="all", dry_run=False):
    if dry_run:
        print("Fixed-model matched diagnostics: 4 stations x 3 backends x 2 splits x (real + 5 replacements).")
        print("Training: 4 stations x clean/noise/state, 100 epochs each; validation only; no test promotion.")
        return
    config = load_experiment_config(ev.ROOT / "configs/formal_experiment.json")
    records = load_best_parameters(config.best_parameters)
    freeze_protocol(config)
    _seed_everything(2026)
    if action in ("diagnose", "all"):
        diagnose(config, records)
    if action in ("train", "all"):
        train(config, records)
    print("COMPLETE", flush=True)
