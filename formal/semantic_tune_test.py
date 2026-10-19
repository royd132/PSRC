"""Explicit test evaluation of validation-locked semantic tuning checkpoints."""
import hashlib
import json

import torch

from formal.config import JobSpec, load_experiment_config
from formal.data import load_dataset
from formal.engine import _model_config, _loader, _evaluate, save_prediction_artifacts, _seed_everything
from formal.semantic_tune import ROOT, OUT
from formal.semantic_study import write_json
from models.factory import build_model


def main():
    config = load_experiment_config(ROOT / "configs/formal_experiment.json")
    protocol = json.loads((OUT / "protocol.json").read_text())
    for name, expected in protocol["sha256"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Campaign fingerprint changed: {name}")
    jobs = []
    for dataset in config.datasets:
        selection = json.loads((OUT / dataset.name / "selection_report.json").read_text())
        for arm in dict.fromkeys(("incumbent", selection["screen_winner"])):
            directory = OUT / dataset.name / arm / "seed_2026"
            lock = json.loads((directory / "lock.json").read_text())
            assert lock["seed"] == 2026
            if arm != "incumbent":
                assert lock["params"] == selection["params"]
            assert (directory / lock["checkpoint"]).is_file()
            jobs.append((dataset, arm, directory, lock))
    rows = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for dataset, arm, directory, lock in jobs:
        checkpoint = directory / lock["checkpoint"]
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        result_path = directory / "test.json"
        if result_path.exists():
            result = json.loads(result_path.read_text())
            assert result["checkpoint_sha256"] == digest
        else:
            print(f"[test] {dataset.name}/{arm}/seed2026", flush=True)
            _seed_everything(2026)
            bundle = load_dataset(dataset.path, 96, 16, config.train_fraction, config.validation_end_fraction)
            job = JobSpec(dataset.name, "psrc", 96, 16, 2026)
            model = build_model("psrc", _model_config(job, bundle, lock["params"], bool(config.psrc["gate"]))).to(device)
            model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)["state_dict"], strict=True)
            loader = _loader(bundle, bundle.test.starts, lock["batch"], False, 2026)
            metrics, arrays = _evaluate(model, loader, bundle, device, collect=True)
            artifact = save_prediction_artifacts(directory, arrays, loader, 96, bundle, metrics)
            result = {"station": dataset.name, "arm": arm, "seed": 2026,
                      "test": metrics, "checkpoint_sha256": digest,
                      "predictions_artifact": artifact.name, "evaluation": "single_locked_test_pass"}
            write_json(result_path, result)
            del model
            torch.cuda.empty_cache()
        rows.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    write_json(OUT / "test_comparison.json", rows)
    print("[complete]", OUT / "test_comparison.json", flush=True)


if __name__ == "__main__":
    main()
