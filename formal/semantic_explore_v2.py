"""Fixed seed2026 optimization/loss exploration; preserves all older campaigns."""
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
PREV = ROOT / "outputs/semantic_explore_physical_v1"
OUT = ROOT / "outputs/semantic_explore_physical_v2"
STATIONS = ("pvod_station02", "dkasc_site1a", "hkust", "dkasc_site31")


def candidates(base):
    changes = {
        "plain_mse": dict(loss_base="mse", daylight_weight=0., ramp_weight=0.),
        **{f"huber_{delta:g}": dict(loss_base="huber", huber_delta=delta)
           for delta in (.25, .5, 1.)},
        "batch_half": dict(batch=max(16, int(base["batch"]) // 2)),
        "batch_double": dict(batch=int(base["batch"]) * 2),
        "optimizer_switch": dict(optimizer="adamw" if base["optimizer"] == "adam" else "adam"),
        "constant_lr": dict(scheduler="none"),
    }
    return {name: {**base, **delta} for name, delta in changes.items()}


def main(dry_run=False, *, source=PREV, output=OUT, stations=STATIONS,
         grid_factory=candidates, study="semantic_explore_physical_v2", extra_sources=()):
    PREV, OUT, STATIONS = source, output, stations
    config = load_experiment_config(ROOT / "configs/formal_experiment.json")
    locks, dirs = {}, {}
    record = protocol(config)
    record.update(study=study, seeds=[2026], arms=["physical"],
        status="exploratory_after_test_exposure_not_confirmatory",
        selection="minimum validation standardized MSE including previous validation winner",
        test_policy="evaluate all locked winners after all searches; retain all results")
    record.pop("lr_factors", None)
    record["sha256"]["formal/semantic_explore_v2.py"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for path in extra_sources:
        record["sha256"][str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    grids = {}
    for station in STATIONS:
        selection_path = PREV / station / "selection.json"
        selected = json.loads(selection_path.read_text())
        dirs[station] = Path(selected["directory"]) if "directory" in selected else PREV / station / selected["candidate"]
        lock_path = dirs[station] / "lock.json"
        locks[station] = json.loads(lock_path.read_text())
        grids[station] = grid_factory(locks[station]["params"])
        for path in (selection_path, lock_path):
            record["sha256"][str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    record["candidates"] = grids
    record = json.loads(json.dumps(record))
    print(f"seed2026: {len(STATIONS)} stations, {sum(map(len, grids.values()))} candidates; 100 epochs/patience10; then locked test results", flush=True)
    if dry_run:
        return
    manifest = OUT / "protocol.json"
    if manifest.exists() and json.loads(manifest.read_text()) != record:
        raise RuntimeError("Protocol changed; do not mix campaign versions")
    write_json(manifest, record)
    for station in STATIONS:
        selection_path = OUT / station / "selection.json"
        if selection_path.exists():
            continue
        bundle = load_dataset(config.dataset(station).path, 96, 16,
                              config.train_fraction, config.validation_end_fraction)
        job = JobSpec(station, "psrc", 96, 16, 2026)
        scores = {"incumbent": locks[station]["validation"]["mse_standardized"]}
        for name, params in grids[station].items():
            directory = OUT / station / name
            lock_path = directory / "lock.json"
            if lock_path.exists():
                lock = json.loads(lock_path.read_text())
            else:
                print(f"[train] {station}/{name}", flush=True)
                def progress(epoch, score):
                    print(f"[epoch] {station}/{name} {epoch}: val={score:.8f}", flush=True)
                model, history, stopper, batch = _train_model(job, config, bundle, params,
                    psrc_settings(params, config.psrc), epoch_callback=progress)
                metrics = _evaluate(model, _loader(bundle, bundle.validation.starts, batch, False, 2026),
                                    bundle, next(model.parameters()).device)
                checkpoint = save_checkpoint(directory, model)
                lock = dict(params=params, seed=2026, validation=metrics, history=history,
                            batch=batch, best_epoch=stopper.best_epoch, checkpoint=checkpoint.name)
                write_json(lock_path, lock)
                del model
                gc.collect()
                torch.cuda.empty_cache()
            scores[name] = lock["validation"]["mse_standardized"]
        winner = min(scores, key=scores.get)
        directory = dirs[station] if winner == "incumbent" else OUT / station / winner
        write_json(selection_path, dict(candidate=winner, scores=scores, directory=str(directory)))
        print(f"[selected] {station}/{winner}", flush=True)
    results = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for station in STATIONS:
        selection = json.loads((OUT / station / "selection.json").read_text())
        directory = Path(selection["directory"])
        lock = json.loads((directory / "lock.json").read_text())
        output = OUT / station / "test.json"
        if output.exists():
            result = json.loads(output.read_text())
        elif selection["candidate"] == "incumbent" and ((PREV / station / "test.json").exists() or (directory / "test.json").exists()):
            previous_test = PREV / station / "test.json"
            if not previous_test.exists():
                previous_test = directory / "test.json"
            result = {**json.loads(previous_test.read_text()),
                      "reused_test_source": str(previous_test)}
            write_json(output, result)
        else:
            print(f"[test] {station}/{selection['candidate']}", flush=True)
            _seed_everything(2026)
            bundle = load_dataset(config.dataset(station).path, 96, 16,
                                  config.train_fraction, config.validation_end_fraction)
            job = JobSpec(station, "psrc", 96, 16, 2026)
            model = build_model("psrc", _model_config(job, bundle, lock["params"], bool(config.psrc["gate"]))).to(device)
            model.load_state_dict(torch.load(directory / lock["checkpoint"], map_location=device,
                                            weights_only=True)["state_dict"], strict=True)
            loader = _loader(bundle, bundle.test.starts, lock["batch"], False, 2026)
            metrics, arrays = _evaluate(model, loader, bundle, device, collect=True)
            save_prediction_artifacts(OUT / station, arrays, loader, 96, bundle, metrics)
            result = dict(station=station, candidate=selection["candidate"], seed=2026,
                          test=metrics, params=lock["params"], status=record["status"])
            write_json(output, result)
            del model
            torch.cuda.empty_cache()
        results.append(result)
    write_json(OUT / "test_comparison.json", results)
    print(f"[complete] {OUT}", flush=True)


if __name__ == "__main__":
    import sys
    main(dry_run="--dry-run" in sys.argv)
