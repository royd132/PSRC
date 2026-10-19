"""Validation-selected, paired-seed controls of semantic specificity."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from formal.config import JobSpec, load_experiment_config
from formal.data import load_dataset
from formal.engine import (
    _evaluate, _loader, _model_config, _parameter_record, _train_model,
    psrc_settings, save_checkpoint, save_prediction_artifacts,
)
from models.factory import build_model

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "semantic_specificity_v2"
SEEDS = (2027, 2028, 2029, 2030, 2031)
LR_FACTORS = (0.5, 1.0, 2.0)
ARMS = {
    "physical": {},
    "regroup_4101": {"regroup_seed": 4101},
    "regroup_4102": {"regroup_seed": 4102},
    "regroup_4103": {"regroup_seed": 4103},
    "generic": {"token_statistics": "generic"},
    "shared_retrieval": {"shared_retrieval": True},
    "mlp_residual": {},
    "concat_stats": {},
}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def ablation(arm):
    return {arm} if arm in ("mlp_residual", "concat_stats") else set()


def trial_dir(station, arm, factor, seed):
    return OUT / station / arm / f"lr_{factor:g}" / f"seed_{seed}"


def protocol(config):
    paths = [ROOT / p for p in (
        "formal/semantic_study.py", "formal/engine.py", "formal/config.py",
        "formal/data.py", "layers/physical_semantic.py",
        "layers/residual_corrector.py", "models/ours.py",
        "configs/formal_experiment.json",
    )] + [config.best_parameters] + [d.path for d in config.datasets]
    return {
        "study": "semantic_specificity_v2",
        "status": "supplementary_after_test_exposure",
        "seeds": SEEDS, "arms": ARMS, "lr_factors": LR_FACTORS,
        "epochs": config.epochs, "patience": config.patience,
        "seq_len": 96, "horizon": 16,
        "selection": "lowest mean validation standardized MSE over all five seeds; ties use first LR factor",
        "test_policy": "explicit test phase, all arms reported, no test promotion",
        "sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
    }


def main(action="plan", *, dry_run=False):
    config = load_experiment_config(ROOT / "configs/formal_experiment.json")
    stations = [d.name for d in config.datasets]
    count = len(stations) * len(ARMS) * len(SEEDS)
    print(f"semantic specificity: {stations}; L96->H16; 100ep/patience10")
    print(f"train trials={count * len(LR_FACTORS)}; locked test runs={count}; output={OUT}")
    if dry_run or action == "plan":
        print("train never evaluates test; test requires all validation selections")
        return
    current = protocol(config)
    canonical = json.dumps(current, sort_keys=True)
    fingerprint = hashlib.sha256(canonical.encode()).hexdigest()
    protocol_path = OUT / "protocol.json"
    if protocol_path.exists():
        previous = json.loads(protocol_path.read_text(encoding="utf-8"))
        if previous != json.loads(canonical):
            raise RuntimeError("Protocol/data/code changed: preserve this campaign; use a new version")
    elif action != "train":
        raise RuntimeError("Run train first to lock the campaign")
    else:
        write_json(protocol_path, current)
    if action == "report":
        report(stations)
        return
    if action == "test":
        for station in stations:
            for arm in ARMS:
                selection_path = OUT / station / arm / "selection.json"
                if not selection_path.exists():
                    raise RuntimeError("Complete all validation selections before test")
                selection = json.loads(selection_path.read_text())
                if selection["fingerprint"] != fingerprint:
                    raise RuntimeError("Selection fingerprint mismatch")
                for seed in SEEDS:
                    directory = trial_dir(station, arm, selection["lr_factor"], seed)
                    lock = json.loads((directory / "lock.json").read_text())
                    if lock["fingerprint"] != fingerprint or not (directory / lock["checkpoint"]).exists():
                        raise RuntimeError("Missing or incompatible validation checkpoint")
    for station in stations:
        bundle = load_dataset(config.dataset(station).path, 96, 16,
                              config.train_fraction, config.validation_end_fraction)
        _, base_params = _parameter_record(JobSpec(station, "psrc", 96, 16, SEEDS[0]), config)
        for arm, overrides in ARMS.items():
            selection_path = OUT / station / arm / "selection.json"
            if action == "train":
                if selection_path.exists():
                    print(f"[skip selected] {station}/{arm}", flush=True)
                    continue
                scores = []
                for factor in LR_FACTORS:
                    values = []
                    for seed in SEEDS:
                        directory = trial_dir(station, arm, factor, seed)
                        lock_path = directory / "lock.json"
                        if lock_path.exists():
                            lock = json.loads(lock_path.read_text())
                            if lock["fingerprint"] != fingerprint:
                                raise RuntimeError("Trial fingerprint mismatch")
                        else:
                            job = JobSpec(station, "psrc", 96, 16, seed)
                            params = {**base_params, **overrides, "lr": float(base_params["lr"]) * factor}
                            settings = psrc_settings(params, config.psrc)
                            print(f"[train] {station}/{arm}/lr{factor}/seed{seed}", flush=True)
                            model, history, stopper, batch = _train_model(
                                job, config, bundle, params, settings, ablation=ablation(arm))
                            validation = _evaluate(model, _loader(bundle, bundle.validation.starts,
                                                   batch, False, seed), bundle, next(model.parameters()).device)
                            checkpoint = save_checkpoint(directory, model)
                            lock = {"fingerprint": fingerprint, "params": params,
                                    "seed": seed, "batch": batch, "checkpoint": checkpoint.name,
                                    "best_epoch": stopper.best_epoch, "history": history,
                                    "validation": validation,
                                    "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}
                            write_json(lock_path, lock)
                            del model
                            torch.cuda.empty_cache()
                        values.append(lock["validation"]["mse_standardized"])
                    scores.append(float(np.mean(values)))
                best = int(np.argmin(scores))
                write_json(selection_path, {"fingerprint": fingerprint,
                           "lr_factor": LR_FACTORS[best], "mean_validation_mse": scores})
            else:
                selection = json.loads(selection_path.read_text())
                if selection["fingerprint"] != fingerprint:
                    raise RuntimeError("Selection fingerprint mismatch")
                for seed in SEEDS:
                    directory = trial_dir(station, arm, selection["lr_factor"], seed)
                    result_path = directory / "test.json"
                    if result_path.exists():
                        print(f"[skip test] {station}/{arm}/seed{seed}", flush=True)
                        continue
                    lock = json.loads((directory / "lock.json").read_text())
                    job = JobSpec(station, "psrc", 96, 16, seed)
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    model = build_model("psrc", _model_config(job, bundle, lock["params"],
                                        bool(config.psrc["gate"]), ablation=ablation(arm))).to(device)
                    state = torch.load(directory / lock["checkpoint"], map_location=device, weights_only=True)
                    model.load_state_dict(state["state_dict"], strict=True)
                    loader = _loader(bundle, bundle.test.starts, lock["batch"], False, seed)
                    metrics, arrays = _evaluate(model, loader, bundle, device, collect=True)
                    save_prediction_artifacts(directory, arrays, loader, 96, bundle, metrics)
                    write_json(result_path, {"fingerprint": fingerprint, "test": metrics,
                               "evaluation": "single_locked_test_pass"})
                    del model
                    torch.cuda.empty_cache()
    if action == "test":
        report(stations)


def report(stations):
    rows = []
    for station in stations:
        for arm in ARMS:
            selection = json.loads((OUT / station / arm / "selection.json").read_text())
            for seed in SEEDS:
                directory = trial_dir(station, arm, selection["lr_factor"], seed)
                result = json.loads((directory / "test.json").read_text())
                rows.append({"station": station, "arm": arm, "seed": seed,
                             "lr_factor": selection["lr_factor"], **result["test"]})
    write_json(OUT / "comparison.json", rows)
    with (OUT / "comparison.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for station in stations:
        baseline = [r for r in rows if r["station"] == station and r["arm"] == "physical"]
        for arm in ARMS:
            group = [r for r in rows if r["station"] == station and r["arm"] == arm]
            row = {"station": station, "arm": arm}
            for key in ("rmse_physical", "mae_physical", "mbe_physical", "r2"):
                if key not in group[0]:
                    continue
                values = np.array([r[key] for r in group])
                row[key + "_mean"] = float(values.mean())
                row[key + "_std"] = float(values.std(ddof=1))
            delta = [(r["rmse_physical"] / b["rmse_physical"] - 1) * 100 for r, b in zip(group, baseline)]
            row["paired_rmse_delta_percent_mean"] = float(np.mean(delta))
            row["physical_wins_seeds"] = sum(d > 0 for d in delta)
            summary.append(row)
    write_json(OUT / "summary.json", summary)
    with (OUT / "summary.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(f"Saved comparison.csv/comparison.json/summary.json to {OUT}")
