"""Frozen Chronos prior calibration and numerical adapter training."""
from __future__ import annotations
import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from formal.chronos_prior import (
    CHRONOS2_QUANTILES,
    build_chronos2_grid_cache,
    contract_prior,
    load_grid_cache,
    prior_cache_path_v4,
    readout,
    scale_disagreement,
    scale_disagreement_segments,
    skew_readout,
)
from formal.config import (
    JobSpec,
    load_best_parameters,
    load_experiment_config,
)
from formal.data import load_dataset
from formal.engine import (
    _evaluate,
    _loader,
    _model_config,
    _train_model,
    build_model,
    psrc_settings,
)
from formal.metrics import regression_metrics
from formal.pc_fra import (
    PcFraPack,
    build_pack,
    compute_epsilon,
    stratified_partners,
    window_scalar_features,
)

SEQ_LEN, HORIZON, SEED = 96, 16, 2026
SHUFFLE_SEED = 2026
HIDDEN, DROPOUT = 96, 0.1
EPOCHS, PATIENCE = 100, 10
LOCKED_CONTEXT = 96
LOCKED_NORM = "raw"

def _checkpoint(station: str) -> Path:
    return Path("outputs/final_h16") / (
        f"{station}_L{SEQ_LEN}_H{HORIZON}_psrc_seed{SEED}"
    ) / "checkpoint.pt"


def _load_bundle(station, config):
    return load_dataset(
        config.dataset(station).path,
        SEQ_LEN,
        HORIZON,
        config.train_fraction,
        config.validation_end_fraction,
    )


def _build_frozen(station, config, bundle, params, state, device):
    job = JobSpec(station, "psrc", SEQ_LEN, HORIZON, SEED)
    model = build_model(
        job.model,
        _model_config(job, bundle, params, bool(config.psrc["gate"])),
    ).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, job


def _infer(model, bundle, starts, batch, device):
    loader = _loader(bundle, starts, batch, False, SEED)
    return _evaluate(model, loader, bundle, device, collect=True)


def _calibrated_lookup(name, scale, q_grid, taus, bundle, starts, psrc):
    """Physical-unit calibrated Chronos trajectory keyed by window start."""
    if name.startswith("skew_b"):
        base_all = skew_readout(q_grid, taus, float(name[len("skew_b"):]))
    else:
        base_all = readout(q_grid, taus, name)
    base_all = contract_prior(base_all, bundle, starts, SEQ_LEN, HORIZON)
    kind, value = scale
    lookup = {}
    for i, start in enumerate(starts):
        prior = base_all[i]
        psrc_prediction = psrc["pred_phys"][i]
        if kind == "scalar":
            calibrated = scale_disagreement(
                prior[None], psrc_prediction[None], float(value)
            )[0]
        else:
            calibrated = scale_disagreement_segments(
                prior[None],
                psrc_prediction[None],
                tuple(float(gamma) for gamma in value),
            )[0]
        lookup[int(start)] = calibrated.astype(np.float32)
    return lookup


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8")


def _merge_packs(train_pack, val_pack):
    return PcFraPack(
        np.concatenate([train_pack.starts, val_pack.starts]),
        np.concatenate([train_pack.embeddings, val_pack.embeddings]),
        np.concatenate([train_pack.partner_starts, val_pack.partner_starts]),
        HORIZON,
    )


def _station_cache(station, config):
    bundle = _load_bundle(station, config)
    path = prior_cache_path_v4(
        bundle.path, HORIZON, LOCKED_CONTEXT, LOCKED_NORM
    )
    starts, q_grid, _ = load_grid_cache(path)
    return bundle, {int(s): i for i, s in enumerate(starts)}, q_grid


def _shuffle_lookup(lookup, seed: int):
    """Permute prior trajectories across windows (marginal-preserving)."""
    keys = sorted(lookup)
    perm = np.random.default_rng(seed).permutation(len(keys))
    return {k: lookup[keys[perm[i]]] for i, k in enumerate(keys)}


def train_one(
    station, config, records, cand_name, scale, tag, out_root: Path,
    *, lr: float, weight_decay: float, epsilon: float,
    huber_beta_c: float = 1.0, lambda_energy: float = 0.0,
    lambda_smooth: float = 0.0, shuffle_prior: bool = False,
    skip_if_done: bool = True,
):
    """Train one PARA adapter; return validation arrays + metrics payload."""
    import dataclasses
    import time

    out_dir = out_root / tag / station
    done = out_dir / "validation.json"
    arrays_path = out_dir / "validation_arrays.npz"
    if skip_if_done and done.is_file() and arrays_path.is_file():
        payload = json.loads(done.read_text(encoding="utf-8"))
        arrays = dict(np.load(arrays_path))
        return payload, arrays

    bundle, grid_lookup, q_grid = _station_cache(station, config)
    params = dict(records[station]["params"])
    state = torch.load(
        _checkpoint(station), map_location="cpu", weights_only=True
    )["state_dict"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frozen, job = _build_frozen(station, config, bundle, params, state, device)
    batch = int(params.get("batch", 128))
    taus = np.asarray(CHRONOS2_QUANTILES)

    train_starts = bundle.train.starts
    val_starts = bundle.validation.starts
    train_a = _infer(frozen, bundle, train_starts, batch, device)[1]
    val_a = _infer(frozen, bundle, val_starts, batch, device)[1]
    sd, mu = float(bundle.feature_sd[-1]), float(bundle.feature_mu[-1])

    def phys(a):
        return a["pred_std"] * sd + mu

    train_q = q_grid[[grid_lookup[int(s)] for s in train_starts]]
    val_q = q_grid[[grid_lookup[int(s)] for s in val_starts]]
    median_lookup = _calibrated_lookup(
        cand_name, scale, train_q, taus, bundle, train_starts,
        {"pred_phys": phys(train_a)},
    )
    median_lookup.update(_calibrated_lookup(
        cand_name, scale, val_q, taus, bundle, val_starts,
        {"pred_phys": phys(val_a)},
    ))
    if shuffle_prior:
        train_lookup = _shuffle_lookup(
            {int(s): median_lookup[int(s)] for s in train_starts}, SEED,
        )
        val_lookup = _shuffle_lookup(
            {int(s): median_lookup[int(s)] for s in val_starts}, SEED + 1,
        )
        median_lookup = {**train_lookup, **val_lookup}
    psrc_lookup = {
        int(s): train_a["pred_std"][i] for i, s in enumerate(train_starts)
    }
    psrc_lookup.update({
        int(s): val_a["pred_std"][i] for i, s in enumerate(val_starts)
    })

    levels, sigmas = [], []
    for s in train_starts:
        lr_, sg, _, _, _, _ = window_scalar_features(
            bundle, int(s), SEQ_LEN, HORIZON
        )
        levels.append(lr_)
        sigmas.append(sg)
    stats = {
        "level_mu": float(np.mean(levels)),
        "level_sd": float(np.std(levels) + 1e-8),
        "sigma_mu": float(np.mean(sigmas)),
        "sigma_sd": float(np.std(sigmas) + 1e-8),
    }
    cfg = {
        "horizon": HORIZON, "variant": "para", "hidden": HIDDEN,
        "dropout": DROPOUT, "film_bound": 0.1, "epsilon": float(epsilon),
        "huber_beta": float(huber_beta_c),
        "energy_weight": float(lambda_energy),
        "smooth_weight": float(lambda_smooth),
        **{k: float(v) for k, v in stats.items()},
    }
    train_partners, _ = stratified_partners(
        train_starts, bundle, SEQ_LEN, HORIZON, SHUFFLE_SEED
    )
    val_partners, _ = stratified_partners(
        val_starts, bundle, SEQ_LEN, HORIZON, SHUFFLE_SEED + 1
    )
    train_pack = build_pack(
        bundle, train_starts, SEQ_LEN, HORIZON,
        median_lookup, psrc_lookup, train_partners,
    )
    val_pack = build_pack(
        bundle, val_starts, SEQ_LEN, HORIZON,
        median_lookup, psrc_lookup, val_partners,
    )
    run_params = copy.deepcopy(params)
    run_params.update({"lr": float(lr), "weight_decay": float(weight_decay),
                       "scheduler": "none"})
    started = __import__("time").perf_counter()
    model, history, stopper, _ = _train_model(
        job, config, bundle, run_params,
        psrc_settings(params, config.psrc),
        epochs=EPOCHS, patience=PATIENCE,
        fm_embeddings=_merge_packs(train_pack, val_pack),
        initial_state_dict=state, train_pc_only=True, pc_fra_config=cfg,
    )
    _, arrays_b = _evaluate(
        model,
        _loader(bundle, val_starts, batch, False, SEED, fm_embeddings=val_pack),
        bundle, device, collect=True,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.pc_fra_adapter.state_dict()},
               out_dir / "adapter_checkpoint.pt")

    target_phys = arrays_b["target_std"] * sd + mu
    pred_b = np.maximum(0.0, arrays_b["pred_std"] * sd + mu)
    pred_a = np.maximum(0.0, val_a["pred_std"] * sd + mu)
    arrays = {
        "psrc_std": val_a["pred_std"].astype(np.float32),
        "pred_b_std": arrays_b["pred_std"].astype(np.float32),
        "target_std": arrays_b["target_std"].astype(np.float32),
        "target_sd": np.float32(sd), "target_mu": np.float32(mu),
    }
    np.savez(arrays_path, **arrays)
    ma = regression_metrics(pred_a, target_phys, bundle.capacity)
    mb = regression_metrics(pred_b, target_phys, bundle.capacity)
    resid_std = float(np.std(
        train_a["pred_std"] - train_a["target_std"]
    ))
    payload = {
        "station": station, "tag": tag,
        "locked_prior": {"candidate": cand_name, "scale": list(scale)},
        "shuffled_prior": bool(shuffle_prior),
        "hyperparameters": {
            "lr": float(lr), "weight_decay": float(weight_decay),
            "epsilon": float(epsilon), "huber_beta_std": float(huber_beta_c),
            "lambda_energy": float(lambda_energy),
            "lambda_smooth": float(lambda_smooth),
            "hidden": HIDDEN, "dropout": DROPOUT,
        },
        "train_residual_std": resid_std,
        "best_epoch": stopper.best_epoch, "epochs_ran": len(history),
        "seconds": __import__("time").perf_counter() - started,
        "A_frozen_psrc": {
            "rmse_physical": float(ma["rmse_physical"]),
            "mae_physical": float(ma["mae_physical"]),
            "mbe_physical": float(ma["mbe_physical"]),
            "r2": float(ma["r2"]),
        },
        "B_adapter": {
            "rmse_physical": float(mb["rmse_physical"]),
            "mae_physical": float(mb["mae_physical"]),
            "mbe_physical": float(mb["mbe_physical"]),
            "r2": float(mb["r2"]),
        },
    }
    _write_json(done, payload)
    return payload, arrays


def _rebuild_model(station, config, records, bundle, state, lock, device):
    """Reconstruct the exact locked model and load the saved adapter."""
    params = dict(records[station]["params"])
    job = JobSpec(station, "psrc", SEQ_LEN, HORIZON, SEED)
    levels, sigmas = [], []
    for s in bundle.train.starts:
        lr_, sg, _, _, _, _ = window_scalar_features(
            bundle, int(s), SEQ_LEN, HORIZON
        )
        levels.append(lr_)
        sigmas.append(sg)
    cfg = {
        "horizon": HORIZON, "variant": "para", "hidden": HIDDEN,
        "dropout": DROPOUT, "film_bound": 0.1,
        "epsilon": float(lock["epsilon"]),
        "huber_beta": float(lock["huber_beta_std"]),
        "energy_weight": float(lock["lambda_energy"]),
        "smooth_weight": float(lock["lambda_smooth"]),
        "level_mu": float(np.mean(levels)),
        "level_sd": float(np.std(levels) + 1e-8),
        "sigma_mu": float(np.mean(sigmas)),
        "sigma_sd": float(np.std(sigmas) + 1e-8),
    }
    model = build_model(
        job.model,
        _model_config(job, bundle, params, True, pc_fra=cfg),
    ).to(device)
    incompatible = model.load_state_dict(state, strict=False)
    missing = [
        n for n in incompatible.missing_keys
        if not n.startswith("pc_fra_adapter.")
    ]
    if incompatible.unexpected_keys or missing:
        raise ValueError(f"frozen warm-start mismatch: {missing}")
    return model, cfg

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--station', required=True)
    parser.add_argument('--config', type=Path, default=Path('configs/formal_experiment.json'))
    parser.add_argument('--settings', type=Path, default=Path('configs/pfa.json'))
    parser.add_argument('--output', type=Path, default=Path('outputs/foundation'))
    args = parser.parse_args()
    config = load_experiment_config(args.config)
    settings = json.loads(args.settings.read_text())['stations'][args.station]['numeric']
    bundle = _load_bundle(args.station, config)
    starts = np.concatenate([bundle.train.starts, bundle.validation.starts])
    build_chronos2_grid_cache(bundle, starts, HORIZON, LOCKED_CONTEXT, LOCKED_NORM)
    hp = settings
    train_one(args.station, config, load_best_parameters(config.best_parameters),
        hp['prior']['candidate'], hp['prior']['scale'], 'selected', args.output,
        lr=hp['lr'], weight_decay=hp['weight_decay'], epsilon=hp['epsilon'],
        huber_beta_c=hp['huber_beta_std'], lambda_energy=hp['lambda_energy'],
        lambda_smooth=hp['lambda_smooth'])


if __name__ == '__main__':
    main()
