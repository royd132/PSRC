"""Equal-budget single-seed semantic controls; validation search then locked test."""
import gc
import hashlib
import json
from pathlib import Path

import optuna
import torch

from formal.config import JobSpec, load_experiment_config
from formal.data import load_dataset
from formal.engine import (_train_model, _evaluate, _loader, _model_config,
                           _seed_everything, psrc_settings, save_checkpoint,
                           save_prediction_artifacts)
from formal.semantic_study import protocol, write_json, ablation
from models.factory import build_model

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/semantic_optuna_seed2026_v1"
TRIALS = 12
ARMS = {"physical": {}, "regroup": {"regroup_seed": 4101},
        "generic": {"token_statistics": "generic"},
        "concat_stats": {}, "mlp_residual": {}}
STATIONS = ("pvod_station02", "dkasc_site1a", "hkust", "dkasc_site31")


def suggest(trial, base):
    return {**base,
        "lr": trial.suggest_float("lr", float(base["lr"]) / 4, float(base["lr"]) * 2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-8, 1e-3, log=True),
        "dropout": trial.suggest_categorical("dropout", [0., .05, .1, .2]),
        "max_correction": trial.suggest_categorical("max_correction", [.125, .25, .5]),
        "max_gate": trial.suggest_categorical("max_gate", [.25, .5, 1.]),
        "semantic_loss_weight": trial.suggest_categorical("semantic_loss_weight", [0., .05, .1, .2, .3, .5]),
        "base_anchor_weight": trial.suggest_categorical("base_anchor_weight", [0., .25, .5, 1., 2.]),
        "residual_alpha": trial.suggest_categorical("residual_alpha", [0., 1., 5.])}


def main(*, dry_run=False):
    if dry_run:
        print("seed2026: 4 stations x 5 arms x 12 trials = 240; 100ep/patience10; then 20 locked test passes")
        return
    config = load_experiment_config(ROOT / "configs/formal_experiment.json")
    record = protocol(config)
    record.update(study="semantic_optuna_seed2026_v1", seeds=[2026], arms=ARMS,
                  trials_per_arm=TRIALS, selection="minimum validation standardized MSE",
                  optuna_version=optuna.__version__, test_policy="test all 20 validation winners after all selections; no test reselection")
    record.pop("lr_factors", None)
    record["sha256"]["formal/semantic_optuna.py"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    bases = {}
    for station in STATIONS:
        path = ROOT / "outputs/semantic_tune_v3_seed2026" / station / "selection_report.json"
        record["sha256"][str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        bases[station] = json.loads(path.read_text())["params"]
    record["bases"] = bases
    record = json.loads(json.dumps(record))
    manifest = OUT / "protocol.json"
    if manifest.exists() and json.loads(manifest.read_text()) != record:
        raise RuntimeError("Protocol changed: preserve campaign and use a new version")
    write_json(manifest, record)
    storage = "sqlite:///" + (OUT / "study.sqlite3").as_posix()
    for station in STATIONS:
        bundle = load_dataset(config.dataset(station).path, 96, 16,
                              config.train_fraction, config.validation_end_fraction)
        job = JobSpec(station, "psrc", 96, 16, 2026)
        for arm, overrides in ARMS.items():
            study = optuna.create_study(study_name=f"{station}_{arm}", storage=storage,
                load_if_exists=True, direction="minimize", sampler=optuna.samplers.TPESampler(seed=2026),
                pruner=optuna.pruners.NopPruner())
            if any(t.state == optuna.trial.TrialState.RUNNING for t in study.trials):
                raise RuntimeError("Interrupted RUNNING trial found; reconcile before resuming")
            if not study.trials:
                base = bases[station]
                study.enqueue_trial({k: base[k] for k in ("lr", "weight_decay", "dropout", "max_correction",
                    "max_gate", "semantic_loss_weight", "base_anchor_weight", "residual_alpha")})

            def objective(trial):
                params = {**suggest(trial, bases[station]), **overrides}
                directory = OUT / station / arm / f"trial_{trial.number:03}"
                print(f"[train] {station}/{arm}/trial{trial.number}", flush=True)
                def progress(epoch, score):
                    trial.report(score, epoch)
                    print(f"[epoch] {station}/{arm}/{trial.number} {epoch} val={score:.7f}", flush=True)
                model, history, stopper, batch = _train_model(job, config, bundle, params,
                    psrc_settings(params, config.psrc), ablation=ablation(arm), epoch_callback=progress)
                validation = _evaluate(model, _loader(bundle, bundle.validation.starts, batch, False, 2026),
                                       bundle, next(model.parameters()).device)
                checkpoint = save_checkpoint(directory, model)
                write_json(directory / "lock.json", {"params": params, "seed": 2026,
                    "validation": validation, "history": history, "best_epoch": stopper.best_epoch,
                    "batch": batch, "checkpoint": checkpoint.name,
                    "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)})
                del model
                gc.collect()
                torch.cuda.empty_cache()
                return float(validation["mse_standardized"])

            finished = sum(t.state.is_finished() for t in study.trials)
            study.optimize(objective, n_trials=max(0, TRIALS-finished), gc_after_trial=True)
            write_json(OUT / station / arm / "selection.json", {"trial": study.best_trial.number,
                       "validation_mse": study.best_value, "seed": 2026})
    # All searches finish before any test metric is computed.
    rows = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for station in STATIONS:
        bundle = load_dataset(config.dataset(station).path, 96, 16,
                              config.train_fraction, config.validation_end_fraction)
        job = JobSpec(station, "psrc", 96, 16, 2026)
        for arm in ARMS:
            selected = json.loads((OUT / station / arm / "selection.json").read_text())
            directory = OUT / station / arm / f"trial_{selected['trial']:03}"
            lock = json.loads((directory / "lock.json").read_text())
            result_path = directory / "test.json"
            if result_path.exists():
                result = json.loads(result_path.read_text())
            else:
                print(f"[test] {station}/{arm}", flush=True)
                _seed_everything(2026)
                model = build_model("psrc", _model_config(job, bundle, lock["params"],
                    bool(config.psrc["gate"]), ablation=ablation(arm))).to(device)
                model.load_state_dict(torch.load(directory / lock["checkpoint"],
                    map_location=device, weights_only=True)["state_dict"], strict=True)
                loader = _loader(bundle, bundle.test.starts, lock["batch"], False, 2026)
                metrics, arrays = _evaluate(model, loader, bundle, device, collect=True)
                save_prediction_artifacts(directory, arrays, loader, 96, bundle, metrics)
                result = {"station": station, "arm": arm, "trial": selected["trial"],
                          "seed": 2026, "test": metrics, "params": lock["params"]}
                write_json(result_path, result)
                del model
                torch.cuda.empty_cache()
            rows.append(result)
    write_json(OUT / "test_comparison.json", rows)
    print(f"[complete] {OUT / 'test_comparison.json'}", flush=True)
