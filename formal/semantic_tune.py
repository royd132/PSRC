"""Bounded validation-only refinement of the physical semantic branch."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from formal.config import JobSpec, load_experiment_config
from formal.data import load_dataset
from formal.engine import _train_model, _evaluate, _loader, psrc_settings, save_checkpoint
from formal.semantic_study import protocol, write_json

SEEDS = (2026,)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/semantic_specificity_v2"
OUT = ROOT / "outputs/semantic_tune_v3_seed2026"
CANDIDATES = {
    "incumbent": {},
    "lr_half": {"lr": .5},
    "residual_half": {"max_correction": .5},
    "residual_double": {"max_correction": 2.},
    "semantic_half": {"semantic_loss_weight": .5},
    "semantic_double": {"semantic_loss_weight": 2.},
    "anchor_half": {"base_anchor_weight": .5},
    "anchor_double": {"base_anchor_weight": 2.},
}


def parameters(base, candidate):
    result = dict(base)
    for key, factor in CANDIDATES[candidate].items():
        result[key] = float(base[key]) * factor
    return result


def main(*, dry_run=False):
    config = load_experiment_config(ROOT / "configs/formal_experiment.json")
    stations = ("pvod_station02", "dkasc_site1a", "hkust", "dkasc_site31")
    if dry_run:
        print("Validation only: (7 candidates + incumbent) x seed2026 x 4 stations = 32 runs;")
        print("no multi-seed confirmation; 100 epochs/patience10; no test evaluation.")
        return
    record = protocol(config)
    record.update(study="semantic_tune_v3_seed2026", candidates=CANDIDATES,
                  seeds=list(SEEDS), screening_seeds=list(SEEDS), confirmation_seeds=[],
                  selection="single seed2026 validation MSE; report MAE separately; no test use")
    record.pop("arms", None)
    record.pop("lr_factors", None)
    record["base_configuration_source"] = "previous validation-selected physical configuration; historical selection used five seeds"
    record["sha256"]["formal/semantic_tune.py"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    bases, sources = {}, {}
    for station in stations:
        directory = SOURCE / station / "physical"
        selection = json.loads((directory / "selection.json").read_text())
        sources[station] = directory / f"lr_{selection['lr_factor']:g}"
        # Reuse only the selected hyperparameters, never another seed's metrics.
        bases[station] = json.loads((sources[station] / "seed_2027" / "lock.json").read_text())["params"]
        for seed in (2027,):
            path = sources[station] / f"seed_{seed}" / "lock.json"
            record["sha256"][str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    record["base_params"] = bases
    record = json.loads(json.dumps(record))
    manifest = OUT / "protocol.json"
    if manifest.exists() and json.loads(manifest.read_text()) != record:
        raise RuntimeError("Campaign source/config changed; do not mix runs")
    write_json(manifest, record)
    for station in stations:
        print(f"[station] {station}", flush=True)
        bundle = load_dataset(config.dataset(station).path, 96, 16,
                              config.train_fraction, config.validation_end_fraction)

        def run(candidate, seed):
            directory = OUT / station / candidate / f"seed_{seed}"
            lock_path = directory / "lock.json"
            if lock_path.exists():
                return json.loads(lock_path.read_text())
            params = parameters(bases[station], candidate)
            job = JobSpec(station, "psrc", 96, 16, seed)
            print(f"[train] {station}/{candidate}/seed{seed}", flush=True)

            def progress(epoch, score):
                print(f"[epoch] {station}/{candidate}/seed{seed} {epoch}: val_mse={score:.8f}", flush=True)

            model, history, stopper, batch = _train_model(
                job, config, bundle, params, psrc_settings(params, config.psrc),
                epoch_callback=progress)
            validation = _evaluate(model, _loader(bundle, bundle.validation.starts,
                                   batch, False, seed), bundle, next(model.parameters()).device)
            checkpoint = save_checkpoint(directory, model)
            lock = {"params": params, "seed": seed, "validation": validation,
                    "best_epoch": stopper.best_epoch, "history": history,
                    "checkpoint": checkpoint.name, "batch": batch}
            write_json(lock_path, lock)
            del model
            torch.cuda.empty_cache()
            return lock

        scores = {}
        for candidate in CANDIDATES:
            runs = [run(candidate, seed) for seed in SEEDS]
            scores[candidate] = float(np.mean([r["validation"]["mse_standardized"] for r in runs]))
        winner = min(scores, key=scores.get)
        write_json(OUT / station / "screen_selection.json", {"scores": scores, "winner": winner})
        selected = [run(winner, seed) for seed in SEEDS]
        incumbent = [run("incumbent", seed) for seed in SEEDS]
        report = {"station": station, "screen_winner": winner, "params": parameters(bases[station], winner),
                  "split": "validation", "seed": SEEDS[0], "test_evaluated": False, "metrics": {}}
        for metric in ("rmse_physical", "mae_physical", "mse_standardized"):
            a = np.array([r["validation"][metric] for r in selected])
            b = np.array([r["validation"][metric] for r in incumbent])
            report["metrics"][metric] = {"candidate": float(a[0]),
                "incumbent": float(b[0]),
                "paired_change_percent": ((a / b - 1) * 100).tolist()}
        report["improves_validation"] = scores[winner] < scores["incumbent"]
        write_json(OUT / station / "selection_report.json", report)
        print(f"[selected] {station}: {winner}; improves={report['improves_validation']}", flush=True)
    print(f"[complete] {OUT}; validation only; original formal results unchanged", flush=True)
