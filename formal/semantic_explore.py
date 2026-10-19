"""Pre-registered, validation-only exploratory tuning of the physical arm.

Status: exploratory_after_test_exposure_not_confirmatory. The candidate grid is
fully fixed before any training starts and is written into protocol.json.
Selection uses validation standardized MSE only; the single locked test pass
runs only after every station selection is written. Results from this campaign
must not be presented as confirmatory evidence: a superiority claim requires a
fresh, untouched period and an equal-budget rerun of every arm.
"""
import csv
import gc
import hashlib
import json
from pathlib import Path

import torch

from formal.config import JobSpec, load_experiment_config
from formal.data import load_dataset
from formal.engine import (_train_model, _evaluate, _loader, _model_config,
                           _seed_everything, psrc_settings, save_checkpoint,
                           save_prediction_artifacts)
from formal.semantic_study import protocol, write_json
from models.factory import build_model

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "semantic_explore_physical_v1"
PREV = ROOT / "outputs" / "semantic_optuna_seed2026_v1"
SEED = 2026
STATIONS = ("pvod_station02", "dkasc_site1a", "hkust", "dkasc_site31")
DROPOUTS = (0.0, 0.05, 0.1, 0.2)


def incumbent(station):
    selected = json.loads((PREV / station / "physical" / "selection.json").read_text())
    lock_path = PREV / station / "physical" / f"trial_{selected['trial']:03}" / "lock.json"
    return json.loads(lock_path.read_text())["params"], selected, lock_path


def candidate_grid(p):
    """Fixed deterministic grid derived only from the validation-selected incumbent."""
    raw = []

    def add(name, **overrides):
        raw.append((name, {**p, **overrides}))

    add("incumbent")
    for factor in (0.5, 0.7, 1.4, 2.0):
        add(f"lr_x{factor:g}", lr=p["lr"] * factor)
    add("wd_x0.1", weight_decay=max(p["weight_decay"] * 0.1, 1e-9))
    add("wd_x10", weight_decay=min(p["weight_decay"] * 10.0, 1e-3))
    for dropout in DROPOUTS:
        add(f"dropout_{dropout:g}", dropout=dropout)
    add("strong_semantic", semantic_loss_weight=0.5, base_anchor_weight=2.0,
        residual_alpha=5.0, max_correction=0.25, max_gate=0.5)
    add("light_semantic", semantic_loss_weight=0.1, base_anchor_weight=0.5,
        residual_alpha=1.0)
    add("no_anchor", base_anchor_weight=0.0)
    add("no_semantic_loss", semantic_loss_weight=0.0)
    add("big_correction", max_correction=0.5, max_gate=1.0)
    add("small_correction", max_correction=0.125)
    add("gate_025", max_gate=0.25)
    add("joint_low_lr_low_wd", lr=p["lr"] * 0.7,
        weight_decay=max(p["weight_decay"] * 0.1, 1e-9), dropout=0.0)
    add("joint_high_lr_dropout", lr=p["lr"] * 1.4, dropout=0.1)

    seen = set()
    grid = []
    for name, params in raw:
        key = json.dumps(params, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        grid.append({"name": name, "params": params})
    return grid


def build_record(config):
    record = protocol(config)
    record.update(
        study="semantic_explore_physical_v1",
        status="exploratory_after_test_exposure_not_confirmatory",
        seeds=[SEED], arms=["physical"],
        selection="minimum validation standardized MSE on seed 2026; pre-registered fixed grid",
        test_policy="one locked test pass after all selections; no test-driven iteration",
        confirmation_requirement="fresh untouched period and equal-budget rerun of every arm",
    )
    record.pop("lr_factors", None)
    record["sha256"]["formal/semantic_explore.py"] = hashlib.sha256(
        Path(__file__).read_bytes()).hexdigest()
    incumbents, grids = {}, {}
    for station in STATIONS:
        params, selected, lock_path = incumbent(station)
        incumbents[station] = {"trial": selected["trial"],
                               "validation_mse": selected["validation_mse"], "params": params}
        rel = lock_path.relative_to(ROOT)
        record["sha256"][str(rel)] = hashlib.sha256(lock_path.read_bytes()).hexdigest()
        grids[station] = candidate_grid(params)
    record["incumbents"] = incumbents
    record["candidates"] = grids
    return json.loads(json.dumps(record))


def main(action="plan", *, dry_run=False):
    config = load_experiment_config(ROOT / "configs/formal_experiment.json")
    total = sum(len(candidate_grid(incumbent(s)[0])) for s in STATIONS)
    print(f"physical exploratory tuning: {list(STATIONS)}; pre-registered candidates={total}")
    print("validation-only selection; one locked test pass; NOT confirmatory")
    print(f"output={OUT}")
    if dry_run or action == "plan":
        for station in STATIONS:
            print(f"  {station}: {len(candidate_grid(incumbent(station)[0]))} candidates")
        return

    record = build_record(config)
    manifest = OUT / "protocol.json"
    if manifest.exists():
        previous = json.loads(manifest.read_text(encoding="utf-8"))
        if previous != record:
            raise RuntimeError("Protocol changed: preserve this campaign and use a new version")
    elif action != "train":
        raise RuntimeError("Run train first to lock the campaign")
    else:
        write_json(manifest, record)

    if action == "train":
        for station in STATIONS:
            selection_path = OUT / station / "selection.json"
            if selection_path.exists():
                print(f"[skip selected] {station}", flush=True)
                continue
            bundle = load_dataset(config.dataset(station).path, 96, 16,
                                  config.train_fraction, config.validation_end_fraction)
            job = JobSpec(station, "psrc", 96, 16, SEED)
            results = []
            for cand in record["candidates"][station]:
                name, params = cand["name"], cand["params"]
                directory = OUT / station / name
                lock_path = directory / "lock.json"
                if lock_path.exists():
                    lock = json.loads(lock_path.read_text(encoding="utf-8"))
                else:
                    print(f"[train] {station}/{name}", flush=True)
                    model, history, stopper, batch = _train_model(
                        job, config, bundle, params,
                        psrc_settings(params, config.psrc), ablation=set())
                    validation = _evaluate(
                        model, _loader(bundle, bundle.validation.starts, batch, False, SEED),
                        bundle, next(model.parameters()).device)
                    checkpoint = save_checkpoint(directory, model)
                    lock = {"name": name, "params": params, "seed": SEED, "batch": batch,
                            "checkpoint": checkpoint.name, "best_epoch": stopper.best_epoch,
                            "history": history, "validation": validation,
                            "trainable_parameters": sum(
                                p.numel() for p in model.parameters() if p.requires_grad)}
                    write_json(lock_path, lock)
                    del model
                    gc.collect()
                    torch.cuda.empty_cache()
                results.append((name, lock["validation"]["mse_standardized"]))
            best_name, best_value = min(results, key=lambda x: x[1])
            write_json(selection_path, {"candidate": best_name,
                       "validation_mse": best_value,
                       "all_validation_mse": [
                           {"candidate": n, "validation_mse": v} for n, v in results],
                       "seed": SEED})
            print(f"[selected] {station}: {best_name} val_mse={best_value:.7f}", flush=True)
        return

    if action == "test":
        for station in STATIONS:
            selection_path = OUT / station / "selection.json"
            if not selection_path.exists():
                raise RuntimeError("Complete all validation selections before test")
        rows = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for station in STATIONS:
            bundle = load_dataset(config.dataset(station).path, 96, 16,
                                  config.train_fraction, config.validation_end_fraction)
            job = JobSpec(station, "psrc", 96, 16, SEED)
            selected = json.loads((OUT / station / "selection.json").read_text())
            name = selected["candidate"]
            directory = OUT / station / name
            lock = json.loads((directory / "lock.json").read_text(encoding="utf-8"))
            print(f"[test] {station}/{name}", flush=True)
            _seed_everything(SEED)
            model = build_model("psrc", _model_config(job, bundle, lock["params"],
                                bool(config.psrc["gate"]), ablation=set())).to(device)
            model.load_state_dict(torch.load(directory / lock["checkpoint"],
                map_location=device, weights_only=True)["state_dict"], strict=True)
            loader = _loader(bundle, bundle.test.starts, lock["batch"], False, SEED)
            metrics, arrays = _evaluate(model, loader, bundle, device, collect=True)
            save_prediction_artifacts(directory, arrays, loader, 96, bundle, metrics)
            result = {"station": station, "candidate": name, "seed": SEED,
                      "status": "exploratory_after_test_exposure_not_confirmatory",
                      "test": metrics, "validation_mse": selected["validation_mse"],
                      "params": lock["params"]}
            write_json(directory / "test.json", result)
            rows.append(result)
            del model
            gc.collect()
            torch.cuda.empty_cache()
        write_json(OUT / "exploratory_comparison.json", rows)
        _summary(rows)
        print(f"[complete] {OUT / 'exploratory_comparison.json'}", flush=True)


def _summary(rows):
    """Side-by-side table; control numbers are read-only references from the prior study."""
    previous = {r["station"] + "/" + r["arm"]: r for r in
                json.loads((PREV / "test_comparison.json").read_text(encoding="utf-8"))}
    table = []
    for row in rows:
        station = row["station"]
        inc = previous[f"{station}/physical"]
        table.append({
            "station": station, "role": "physical_exploratory_best",
            "candidate": row["candidate"],
            "rmse": row["test"]["rmse_physical"], "mae": row["test"]["mae_physical"],
            "validation_mse": row["validation_mse"], "source": "semantic_explore_physical_v1"})
        table.append({
            "station": station, "role": "physical_incumbent_optuna_v1",
            "candidate": f"trial_{inc['trial']}",
            "rmse": inc["test"]["rmse_physical"], "mae": inc["test"]["mae_physical"],
            "validation_mse": None, "source": "semantic_optuna_seed2026_v1"})
        for arm in ("regroup", "generic", "concat_stats", "mlp_residual"):
            ref = previous[f"{station}/{arm}"]
            table.append({
                "station": station, "role": f"control_{arm}", "candidate": f"trial_{ref['trial']}",
                "rmse": ref["test"]["rmse_physical"], "mae": ref["test"]["mae_physical"],
                "validation_mse": None, "source": "semantic_optuna_seed2026_v1"})
    with (OUT / "exploratory_summary.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    write_json(OUT / "exploratory_summary.json", table)


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "plan")
