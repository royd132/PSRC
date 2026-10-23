"""Single command-line runner for the configured formal comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from formal.config import ExperimentConfig, JobSpec, load_experiment_config
from formal.engine import _parameter_record, run_job

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "formal_experiment.json"
DRIFT_LIMIT = 0.005


@dataclass(frozen=True)
class ExecutionSummary:
    planned: int
    executed: int
    skipped: int


def config_fingerprint(job: JobSpec, config: ExperimentConfig) -> str:
    parameter_source, parameters = _parameter_record(job, config)
    payload = {
        "job": asdict(job),
        "dataset_sha256": hashlib.sha256(
            config.dataset(job.dataset).path.read_bytes()
        ).hexdigest(),
        "epochs": config.epochs,
        "patience": config.patience,
        "train_fraction": config.train_fraction,
        "validation_end_fraction": config.validation_end_fraction,
        "output_projection": config.output_projection,
        "train_windows_per_epoch": config.train_windows_per_epoch,
        "psrc": config.psrc,
        "parameter_source": parameter_source,
        "parameters": parameters,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _artifacts_complete(result_path: Path, result: dict[str, object]) -> bool:
    directory = result_path.parent
    prediction_name = result.get("predictions_artifact")
    if not prediction_name or not (directory / str(prediction_name)).is_file():
        return False
    # Deterministic baselines store no checkpoint; every trained model must.
    checkpoint_name = result.get("checkpoint_artifact")
    if checkpoint_name and not (directory / str(checkpoint_name)).is_file():
        return False
    return True


def should_skip(
    result_path: Path,
    fingerprint: str,
    *,
    require_artifacts: bool = False,
) -> bool:
    if not result_path.is_file():
        return False
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    valid = (
        result.get("status") == "complete"
        and result.get("config_fingerprint") == fingerprint
        and result.get("headline_test_evaluation_passes") == 1
    )
    if not valid:
        return False
    if require_artifacts and not _artifacts_complete(result_path, result):
        return False
    return True


def _backup_legacy_result(result_path: Path) -> None:
    """Preserve the pre-dump archive before the first overwrite of a task."""
    backup_path = result_path.with_name("result.predump_backup.json")
    if not backup_path.is_file():
        backup_path.write_text(
            result_path.read_text(encoding="utf-8"), encoding="utf-8"
        )


def _drift_record(
    job: JobSpec,
    old: dict[str, object],
    new: dict[str, object],
    fingerprint: str,
) -> dict[str, object]:
    old_fingerprint = old.get("config_fingerprint")
    record: dict[str, object] = {
        "key": job.key,
        "old_fingerprint": old_fingerprint,
        "new_fingerprint": fingerprint,
    }
    if old_fingerprint != fingerprint:
        record["status"] = "fingerprint_changed"
        return record
    old_rmse = float(old["test"]["rmse_physical"])
    new_rmse = float(new["test"]["rmse_physical"])
    relative = abs(new_rmse - old_rmse) / max(abs(old_rmse), 1e-12)
    record.update(
        {
            "status": "conflict" if relative > DRIFT_LIMIT else "reproduced",
            "old_rmse_physical": old_rmse,
            "new_rmse_physical": new_rmse,
            "relative_abs_diff": relative,
        }
    )
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train the PSRC backbone and comparison models.")
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--station', help='run one configured station')
    parser.add_argument('--model', help='run one configured model, e.g. psrc')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--require-artifacts', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    config = load_experiment_config(args.config)
    jobs = [j for j in config.jobs() if (not args.station or j.dataset == args.station)
            and (not args.model or j.model == args.model)]
    if not jobs:
        parser.error('No matching station/model in the configuration')
    if args.dry_run:
        for job in jobs:
            print(job.key)
        return ExecutionSummary(len(jobs), 0, 0)
    output = (args.output or config.output_dir).resolve()
    executed = skipped = 0
    for job in jobs:
        path = output / job.key / 'result.json'
        fingerprint = config_fingerprint(job, config)
        if args.resume and should_skip(path, fingerprint, require_artifacts=args.require_artifacts):
            skipped += 1
            continue
        if path.exists():
            _backup_legacy_result(path)
        result = run_job(job, config, artifacts_dir=path.parent)
        result['config_fingerprint'] = fingerprint
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        temporary.replace(path)
        executed += 1
    summary = ExecutionSummary(len(jobs), executed, skipped)
    print(summary)
    return summary


if __name__ == '__main__':
    main()
