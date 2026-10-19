"""Supplementary controls after test exposure; never updates the headline table.

Reuse the existing PSRC/adapter engine. Scoped hooks replace only the prior
cache and per-window prior; all backends are retrained under one protocol.
"""
from __future__ import annotations

import csv
import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from formal import fm_campaign as fm
from formal.config import load_best_parameters, load_experiment_config
from layers.pc_fra import pack_slices

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/evidence_study_v1"
STATIONS = ("dkasc_site31", "dkasc_site1a", "pvod_station02", "hkust")
BACKENDS = ("chronos2", "moirai2", "timesfm3")
ARMS = (("chronos2", "none"),) + tuple(
    (b, a) for b in BACKENDS for a in ("real", "shuffle")
)
# Fixed before any supplementary training; beta is in GLOBAL standardized units.
SETTINGS = {
    "dkasc_site31": ("skew_b+0.50", 3e-4, 1e-3, .5, .03238818049430847, .01, .01),
    "dkasc_site1a": ("mean_q_full", 3e-4, 0., 1., 1., 0., 0.),
    "pvod_station02": ("mean_q_full", 3e-4, 0., 1., 1., 0., 0.),
    "hkust": ("mean_q_full", 3e-5, 0., .25, 1., 0., 0.),
}


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def protocol():
    return dict(version=1, status="supplementary_after_test_exposure", seed=2026,
                seq_len=96, horizon=16, epochs=100, patience=10,
                settings=SETTINGS, arms=ARMS, adapter="para", hidden=96,
                prior_gamma=.5, beta_units="global_standardized_target",
                shuffle="permute clipped FM BEFORE mixing, independently within each split",
                no_prior="prior equals unprojected PSRC; disagreement equals zero",
                alpha="validation-only clipped least squares, identical for every arm",
                test="evaluate only after all 28 validation locks are complete",
                quantiles="existing 21-quantile caches; interpolated flat tails for replacements",
                selection="no test-based promotion; report every arm")


def manifest(config):
    value = protocol()
    value["inputs"] = {}
    for s in STATIONS:
        paths = [config.dataset(s).path, fm._checkpoint(s)]
        paths += [cache_path(s, b, config, split) for b in BACKENDS
                  for split in ("dev", "test")]
        value["inputs"][s] = {str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest()
                              for p in paths}
    value["implementation_sha256"] = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in ("formal/evidence_study.py", "formal/fm_campaign.py",
                     "formal/engine.py", "models/ours.py", "layers/pc_fra.py")}
    path = OUT / "protocol.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != json.loads(json.dumps(value)):
        raise ValueError("Study inputs/protocol changed: use a new study version, do not reuse results")
    write_json(path, value)


def cache_path(station, backend, config, split):
    if backend == "chronos2":
        if split == "dev":
            return ROOT / fm.prior_cache_path_v4(config.dataset(station).path, 16, 96, "raw")
        return ROOT / "outputs/correction_tune_v1/test_prior_cache" / f"{station}__h16__ctx96__raw__test_grid.npz"
    base = ROOT / "outputs/prior_replacement_v1" / backend
    if split == "test":
        return base / "test_prior_cache" / f"{station}__h16__ctx96__raw__test_grid.npz"
    digest = hashlib.sha256(config.dataset(station).path.read_bytes()).hexdigest()[:8]
    return base / "dev_grid" / f"{digest}__{backend}__h16__ctx96__raw__prior_v4.npz"


def controlled_prior(raw, psrc, arm, seed):
    """No target values are accepted by this transform."""
    if arm == "none":
        return psrc.copy()
    if arm == "shuffle":
        raw = raw[np.random.default_rng(seed).permutation(len(raw))]
    elif arm != "real":
        raise ValueError(arm)
    return (psrc + .5 * (raw - psrc)).astype(np.float32)


@contextmanager
def prior_hooks(station, backend, arm, config):
    original_lookup, original_pack = fm._calibrated_lookup, fm.build_pack

    def station_cache(s, c):
        bundle = fm._load_bundle(s, c)
        path = cache_path(s, backend, c, "dev")
        with np.load(path) as z:
            actual = str(z["dataset_sha256"])
        if actual != hashlib.sha256(bundle.path.read_bytes()).hexdigest():
            raise ValueError("Prior cache dataset hash mismatch")
        starts, grid, _ = fm.load_grid_cache(path)
        return bundle, {int(s): i for i, s in enumerate(starts)}, grid

    def lookup(name, scale, q_grid, taus, bundle, starts, psrc):
        # Extract the physically constrained FM forecast without blending.
        raw_map = original_lookup(name, ("scalar", 1.), q_grid, taus, bundle, starts, psrc)
        raw = np.stack([raw_map[int(s)] for s in starts])
        train = set(map(int, bundle.train.starts))
        val = set(map(int, bundle.validation.starts))
        keys = set(map(int, starts))
        split_id = 0 if keys <= train else 1 if keys <= val else 2
        if split_id == 2 and not keys <= set(map(int, bundle.test.starts)):
            raise ValueError("Control input crosses split boundaries")
        values = controlled_prior(raw, psrc["pred_phys"], arm, 2026 + split_id)
        return {int(s): values[i] for i, s in enumerate(starts)}

    def pack(bundle, starts, seq_len, horizon, median_lookup, psrc_lookup, partners):
        result = original_pack(bundle, starts, seq_len, horizon, median_lookup, psrc_lookup, partners)
        # build_pack normally re-clips the mixed prediction. Preserve the
        # intended mixed prior; in the no-prior arm this guarantees d == 0.
        result.embeddings[:, pack_slices(horizon)["c_bar"]] = np.stack(
            [median_lookup[int(s)] for s in starts])
        return result

    with patch.object(fm, "_station_cache", station_cache), \
         patch.object(fm, "_calibrated_lookup", lookup), \
         patch.object(fm, "build_pack", pack), \
         patch.object(fm, "EPOCHS", 100), patch.object(fm, "PATIENCE", 10):
        yield


def tag(backend, arm):
    return f"{backend}__{arm}"


def train(config, records):
    sigma_report = json.loads((ROOT / "outputs/correction_tune_v1/round1_report.json").read_text())
    for station in STATIONS:
        cand, lr, wd, c, beta, energy, smooth = SETTINGS[station]
        sigma = sigma_report["per_station"][station]["train_residual_std"]
        for backend, arm in ARMS:
            directory = OUT / tag(backend, arm) / station
            print(f"TRAIN {station} {backend} {arm}", flush=True)
            with prior_hooks(station, backend, arm, config):
                result, arrays = fm.train_one(
                    station, config, records, cand, ("scalar", .5), tag(backend, arm), OUT,
                    lr=lr, weight_decay=wd, epsilon=c * sigma, huber_beta_c=beta,
                    lambda_energy=energy, lambda_smooth=smooth)
            if not np.isclose(result["train_residual_std"], sigma, atol=2e-4):
                raise ValueError("Frozen PSRC training residual changed")
            write_json(directory / "lock.json", dict(
                alpha=fm.shrinkage_analysis(arrays)["alpha_star"],
                epsilon=c*sigma, huber_beta_std=beta, lambda_energy=energy,
                lambda_smooth=smooth, source="validation_only"))


def evaluate(config, records):
    for station in STATIONS:
        for backend, arm in ARMS:
            if not (OUT / tag(backend, arm) / station / "lock.json").exists():
                raise RuntimeError("Complete every validation lock before supplementary test evaluation")
    rows = []
    for station in STATIONS:
        bundle = fm._load_bundle(station, config)
        params = records[station]["params"]
        state = torch.load(fm._checkpoint(station), map_location="cpu", weights_only=True)["state_dict"]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        frozen, _ = fm._build_frozen(station, config, bundle, params, state, device)
        starts = bundle.test.starts
        batch = int(params.get("batch", 128))
        a = fm._infer(frozen, bundle, starts, batch, device)[1]
        sd, mu = float(bundle.feature_sd[-1]), float(bundle.feature_mu[-1])
        psrc, target = a["pred_std"], a["target_std"]
        psrc_phys = psrc*sd+mu
        def metrics(pred):
            return fm._phys_metrics(psrc, pred, target, sd, mu, bundle.capacity)
        baseline = metrics(psrc)
        def add(backend, arm, m):
            rows.append(dict(station=station, backend=backend, arm=arm, **m,
                             delta_rmse_pct=100*(m["rmse_physical"]/baseline["rmse_physical"]-1)))
        add("none", "PSRC", baseline)
        for backend in BACKENDS:
            path = cache_path(station, backend, config, "test")
            cs, grid, _ = fm.load_grid_cache(path)
            index = {int(s): i for i, s in enumerate(cs)}
            q = grid[[index[int(s)] for s in starts]]
            raw_map = fm._calibrated_lookup(SETTINGS[station][0], ("scalar", 1.), q,
                np.asarray(fm.CHRONOS2_QUANTILES), bundle, starts, {"pred_phys": psrc_phys})
            raw = np.stack([raw_map[int(s)] for s in starts])
            add(backend, "FM_physical_constraints_no_PSRC", metrics((raw-mu)/sd))
            add(backend, "fixed_mix_50pct", metrics((controlled_prior(raw, psrc_phys, "real", 0)-mu)/sd))
            for arm in (("none", "real", "shuffle") if backend == "chronos2" else ("real", "shuffle")):
                directory = OUT / tag(backend, arm) / station
                lock = json.loads((directory / "lock.json").read_text())
                result_path = directory / "test.json"
                if result_path.exists():
                    m = json.loads(result_path.read_text())["metrics"]
                else:
                    prior = controlled_prior(raw, psrc_phys, arm, 2028)
                    partners, _ = fm.stratified_partners(starts, bundle, 96, 16, 2028)
                    with prior_hooks(station, backend, arm, config):
                        pack = fm.build_pack(bundle, starts, 96, 16,
                            {int(s): prior[i] for i, s in enumerate(starts)},
                            {int(s): psrc[i] for i, s in enumerate(starts)}, partners)
                    model, _ = fm._rebuild_model(station, config, records, bundle, state, lock, device)
                    model.pc_fra_adapter.load_state_dict(torch.load(directory / "adapter_checkpoint.pt",
                        map_location=device, weights_only=True)["state_dict"])
                    arrays = fm._evaluate(model, fm._loader(bundle, starts, batch, False, 2026,
                        fm_embeddings=pack), bundle, device, collect=True)[1]
                    prediction = psrc + lock["alpha"]*(arrays["pred_std"]-psrc)
                    m = metrics(prediction)
                    np.savez_compressed(directory / "test_arrays.npz", pred_std=prediction,
                        psrc_std=psrc, target_std=target, starts=starts, target_mu=mu, target_sd=sd)
                    write_json(result_path, dict(metrics=m, alpha=lock["alpha"],
                        status="supplementary_after_test_exposure"))
                add("none" if arm == "none" else backend, "adapter_"+arm, m)
    with (OUT / "comparison.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    write_json(OUT / "comparison.json", rows)


def audit():
    source = ROOT / "outputs/ablation_20260920/ablation_table_long.csv"
    with source.open(encoding="utf-8-sig") as f:
        ablations = list(csv.DictReader(f))
    lines = ["# PSRC evidence audit", "", "Existing exposed-test results; descriptive, not independent confirmation.", "",
             "## Physical semantic controls", "",
             "Compare numerical_backbone / numeric_residual / PSRC WITHOUT Chronos. The historical eight-row table's structural arms were trained without FM, so Full Ours is not their matched control.", "",
             "|Station|Numerical RMSE|Numeric residual RMSE|PSRC RMSE|PSRC vs numeric residual MAE|", "|---|---:|---:|---:|---:|"]
    for s in ("Site31", "Site1A", "PVOD02", "HKUST"):
        r = {v["variant"]: v for v in ablations if v["station"] == s}
        base, numeric, full = [r[k] for k in ("w/o PSRC", "w/o Physical States", "w/o Chronos")]
        mae_delta = 100*(float(full["mae_mw"])/float(numeric["mae_mw"])-1)
        lines.append(f"|{s}|{base['rmse_mw']}|{numeric['rmse_mw']}|{full['rmse_mw']}|{mae_delta:+.2f}%|")
    lines += ["", "The legacy column names rmse_mw/mae_mw do not establish physical units; use each dataset's documented native power unit.", "",
              "## Corrections to the prior-replacement interpretation", "",
              "- 0.290406/0.289516 belong to Site31, not Site1A.",
              "- Legacy prior_only means physically constrained FM blended 50% with PSRC.",
              "- Legacy Chronos beta=1 on three stations differs from replacement beta=sigma_R; do not claim a single-factor comparison.",
              "- No-prior and per-backend shuffled controls require new training; the v1 study supplies 28 fixed adapter runs.",
              "- Physical-token shuffle under the matched PSRC-only training protocol remains missing; existing numerical residual is an architectural control, not equal-parameter proof.",
              "- Model interface portability does not establish uniform transfer gains or causal attribution to pretraining.", "",
              "## Paper argument", "",
              "Core: historical physical state conditions horizon-wise residual calibration on a time-frequency multi-scale predictor.",
              "Extension: frozen forecast priors can be exchanged through the same interface; report all metrics and failures against fixed blending, no-prior and shuffled controls.",
              "Prioritize matched PSRC semantic controls and history-defined regime analyses over adding further foundation models.",
              "Do not select configurations/stations on supplementary test results. Future independent time periods are needed for fresh confirmation."]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "evidence_audit.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def main(action="audit", dry_run=False):
    config = load_experiment_config(ROOT / "configs/formal_experiment.json")
    records = load_best_parameters(config.best_parameters)
    audit()
    if dry_run:
        print(json.dumps(protocol(), indent=2)); return
    if action == "audit":
        print(OUT / "evidence_audit.md"); return
    manifest(config)
    if action in ("train", "all"):
        train(config, records)
    if action in ("test", "all"):
        evaluate(config, records)


if __name__ == "__main__":
    main()
