"""Isolated physical-language study: plan/audit/prepare/train/test/report.

Existing checkpoints, caches and headline results are read-only. Test is a
separate explicit action, requiring all 20 validation selections first.
"""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from formal import fm_campaign as fm
from formal.config import load_best_parameters, load_experiment_config
from formal.engine import _seed_everything
from formal.metrics import regression_metrics
from formal.physical_prompts import (MODEL_ID, REVISION, encode_prompts, fit_vocabulary,
                                     numeric_features, prompt_bank, state_codes)
from formal.semantic_study import write_json
from layers.pc_fra import pack_slices
from layers.physical_semantic import compute_physical_tokens
from models.physical_language import PhysicalLanguageCorrection, state_alignment

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/physical_language.json"
ARMS = ("numeric", "language_unaligned", "language_aligned",
        "language_shuffled", "random_lm_aligned")
ALIGNED = {"language_aligned", "language_shuffled", "random_lm_aligned"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def setup():
    cfg = read_json(CONFIG)
    formal = load_experiment_config(ROOT / "configs/formal_experiment.json")
    if cfg["seed"] != 2026 or cfg["epochs"] != 100 or cfg["patience"] != 10:
        raise ValueError("This version locks seed2026 and the 100/10 budget")
    if set(cfg["stations"]) != {s.name for s in formal.datasets}:
        raise ValueError("The four-station set must match the formal study")
    output = (ROOT / cfg["output"]).resolve()
    if output != ROOT / "outputs/physical_language_v1":
        raise ValueError("Refuse output redirection into existing formal results")
    return cfg, formal, output


def dev_cache(formal, station):
    return ROOT / fm.prior_cache_path_v4(formal.dataset(station).path, 16, 96, "raw")


def manifest(cfg, formal, out):
    from huggingface_hub import snapshot_download
    from importlib.metadata import version
    # Never download implicitly; fail with an actionable message if absent.
    snapshot = Path(snapshot_download(MODEL_ID, revision=REVISION, local_files_only=True))
    files = [CONFIG, ROOT / "configs/formal_experiment.json", formal.best_parameters]
    files += [ROOT / p for p in (
        "formal/physical_language.py", "formal/physical_prompts.py", "models/physical_language.py",
        "formal/fm_campaign.py", "formal/chronos_prior.py", "formal/pc_fra.py",
        "formal/engine.py", "formal/data.py", "models/ours.py", "layers/physical_semantic.py",
        "layers/residual_corrector.py", "layers/pc_fra.py")]
    for station, settings in cfg["stations"].items():
        directory = ROOT / settings["adapter_dir"]
        files += [formal.dataset(station).path, ROOT / fm._checkpoint(station),
                  directory / "validation.json", directory / "validation_arrays.npz",
                  directory / "adapter_checkpoint.pt", dev_cache(formal, station)]
    files += [snapshot / f for f in ("config.json", "model.safetensors", "vocab.json", "merges.txt")]
    value = dict(config=cfg, arms=ARMS, pretrained_model=MODEL_ID, revision=REVISION,
                 status="exploratory_after_prior_test_exposure_not_confirmatory",
                 selection="validation projected MSE; MAE guard; identity scale included",
                 test="explicit only, all four stations and five arms locked first",
                 known_limit="Frozen base/adapter training residuals are in-sample, not OOF",
                 software={name: version(name) for name in ("torch", "numpy", "transformers")},
                 hashes={str(p): sha(p) for p in files})
    value = json.loads(json.dumps(value))
    path = out / "protocol.json"
    if path.exists() and read_json(path) != value:
        raise ValueError("Inputs/code changed: preserve this version and create a new protocol")
    write_json(path, value)


def metrics(pred, data):
    if not np.isfinite(pred).all() or not np.isfinite(data["target"]).all():
        raise ValueError("Nonfinite predictions/targets are not silently dropped")
    return regression_metrics(np.maximum(0, pred * data["sd"] + data["mu"]),
                              data["target"] * data["sd"] + data["mu"], float(data["capacity"]))


def frozen_features(station, split, cfg, formal, *, audit_rows=None):
    """Rebuild the EXACT registered old path; caller gates any test access."""
    if split not in ("train", "validation", "test"):
        raise ValueError(split)
    _seed_everything(2026)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = fm._load_bundle(station, formal)
    starts = getattr(bundle, split).starts
    if split == "train":
        starts = starts[np.linspace(0, len(starts)-1, min(cfg["train_windows"], len(starts)), dtype=int)]
    if audit_rows:
        starts = starts[:audit_rows]
    path = dev_cache(formal, station) if split != "test" else ROOT / (
        f"outputs/correction_tune_v1/test_prior_cache/{station}__h16__ctx96__raw__test_grid.npz")
    with np.load(path, allow_pickle=False) as cached:
        if str(cached["dataset_sha256"]) != sha(bundle.path):
            raise ValueError("Chronos cache dataset hash mismatch")
    cached_starts, grid, _ = fm.load_grid_cache(path)
    lookup = {int(s): i for i, s in enumerate(cached_starts)}
    if any(int(s) not in lookup for s in starts):
        raise ValueError("Missing Chronos windows; this study never regenerates old caches")
    records = load_best_parameters(formal.best_parameters)
    state = torch.load(ROOT / fm._checkpoint(station), map_location="cpu", weights_only=True)["state_dict"]
    model, _ = fm._build_frozen(station, formal, bundle, records[station]["params"], state, device)
    batch = int(records[station]["params"]["batch"])
    psrc = fm._infer(model, bundle, starts, batch, device)[1]["pred_std"]
    directory = ROOT / cfg["stations"][station]["adapter_dir"]
    registration = read_json(directory / "validation.json")
    name = registration["locked_prior"]["candidate"]
    scale = registration["locked_prior"]["scale"]
    sd, mu = float(bundle.feature_sd[-1]), float(bundle.feature_mu[-1])
    prior_lookup = fm._calibrated_lookup(name, scale, grid[[lookup[int(s)] for s in starts]],
        np.asarray(fm.CHRONOS2_QUANTILES), bundle, starts, {"pred_phys": psrc * sd + mu})
    # Partner features are unused in the locked PARA path; identity partners
    # avoid introducing unrelated windows and preserve the relevant pack fields.
    pack = fm.build_pack(bundle, starts, 96, 16, prior_lookup,
                         {int(s): psrc[i] for i, s in enumerate(starts)}, starts)
    del model
    model, _ = fm._rebuild_model(station, formal, records, bundle, state,
                                  registration["hyperparameters"], device)
    model.pc_fra_adapter.load_state_dict(torch.load(directory / "adapter_checkpoint.pt",
        map_location=device, weights_only=True)["state_dict"], strict=True)
    model.eval().requires_grad_(False)
    arrays = fm._evaluate(model, fm._loader(bundle, starts, batch, False, 2026, fm_embeddings=pack),
                          bundle, device, collect=True)[1]
    alpha = cfg["stations"][station]["alpha"]
    old = psrc + alpha * (arrays["pred_std"] - psrc)
    raw, physical = [], []
    encoder = model.physical_semantic_encoder
    with torch.no_grad():
        for begin in range(0, len(starts), batch):
            ix = starts[begin:begin+batch, None] + np.arange(96)[None]
            x = torch.as_tensor(bundle.values[ix], device=device)
            raw.append(compute_physical_tokens(x, encoder.roles, encoder.feature_mu,
                       encoder.feature_sd, encoder.target_mu, encoder.target_sd).cpu().numpy())
            physical.append(encoder(x).cpu().numpy())
    roles = encoder.roles
    role_valid = [True, roles["clear_sky"] is not None and roles["clear_sky"] == bundle.clear_sky_index,
                  roles["irradiance"] is not None, roles["temperature"] is not None, True, True]
    prior = (pack.embeddings[:, pack_slices(16)["c_bar"]] - mu) / sd
    result = dict(starts=starts, old=old.astype(np.float32), prior=prior.astype(np.float32),
                  disagreement=(prior-psrc).astype(np.float32), physical=np.concatenate(physical),
                  raw=np.concatenate(raw), target=arrays["target_std"].astype(np.float32),
                  mu=np.float64(mu), sd=np.float64(sd), capacity=np.float64(bundle.capacity),
                  role_valid=np.asarray(role_valid))
    if split == "validation":
        with np.load(directory / "validation_arrays.npz", allow_pickle=False) as saved:
            index = np.searchsorted(bundle.validation.starts, starts)
            reference = saved["psrc_std"][index] + alpha * (
                saved["pred_b_std"][index] - saved["psrc_std"][index])
            np.testing.assert_allclose(old, reference, atol=1e-5, rtol=0,
                                       err_msg="Old formal path did not reproduce; stop before training")
            result["old_max_abs_reconstruction_error"] = np.max(np.abs(old-reference))
    del model
    torch.cuda.empty_cache()
    return result


def load_cache(out, station, split):
    with np.load(out / station / f"{split}.npz", allow_pickle=False) as data:
        return dict(data)


def prepare(cfg, formal, out):
    for station in cfg["stations"]:
        folder = out / station
        for split in ("train", "validation"):
            path = folder / f"{split}.npz"
            if not path.exists():
                print(f"[cache] {station}/{split}", flush=True)
                data = frozen_features(station, split, cfg, formal)
                folder.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(path, **data)
        data = load_cache(out, station, "train")
        vocabulary = fit_vocabulary(data["raw"], data["role_valid"])
        vocab_path = folder / "vocabulary.json"
        if vocab_path.exists() and read_json(vocab_path) != vocabulary:
            raise ValueError("Training vocabulary changed")
        write_json(vocab_path, vocabulary)
        prompts, keys = prompt_bank(vocabulary)
        write_json(folder / "prompts.json", dict(prompts=prompts, keys=keys.tolist()))
        for random_weights in (False, True):
            name = "random_bank.npz" if random_weights else "language_bank.npz"
            if not (folder / name).exists():
                print(f"[GPT2 cache] {station}/{name}: {len(prompts)} unique prompts", flush=True)
                embeddings = encode_prompts(prompts, random_weights=random_weights,
                    device="cuda" if torch.cuda.is_available() else "cpu")
                np.savez_compressed(folder / name, keys=keys, embeddings=embeddings)


def shuffled_codes(codes, train_codes, split):
    """Whole-window mismatches; validation/test donors come ONLY from train."""
    rng = np.random.default_rng(2026)
    if split == "train":
        if len(codes) < 2:
            raise ValueError("Shuffling requires at least two training windows")
        order = rng.permutation(len(codes))
        donors = np.empty_like(order)
        donors[order] = np.roll(order, 1)
        return codes[donors]
    if split not in ("validation", "test"):
        raise ValueError(split)
    return train_codes[rng.integers(len(train_codes), size=len(codes))]


def inputs(data, out, station, arm, *, split):
    if arm not in ARMS:
        raise ValueError(arm)
    vocabulary = read_json(out / station / "vocabulary.json")
    codes = state_codes(data["raw"], vocabulary)
    available = np.broadcast_to(np.asarray(vocabulary["active"]).any(1), codes.shape).copy()
    if arm == "numeric":
        representation = numeric_features(data["raw"], vocabulary)
    else:
        donor_codes = codes
        if arm == "language_shuffled":
            training = load_cache(out, station, "train")
            donor_codes = shuffled_codes(codes, state_codes(training["raw"], vocabulary), split)
        bank = "random_bank.npz" if arm == "random_lm_aligned" else "language_bank.npz"
        with np.load(out / station / bank, allow_pickle=False) as z:
            mapping = {tuple(k): i for i, k in enumerate(z["keys"])}
            indices = np.asarray([[mapping[(k, int(c))] for k, c in enumerate(row)] for row in donor_codes])
            representation = z["embeddings"][indices]
    values = [data[k] for k in ("old", "prior", "disagreement", "physical")]
    values += [representation, available, codes, data["target"]]
    return tuple(torch.as_tensor(v, dtype=torch.long if i == 6 else torch.float32)
                 for i, v in enumerate(values))


def predict(model, tensors, device):
    model.eval()
    output = []
    with torch.no_grad():
        for begin in range(0, len(tensors[0]), 512):
            batch = [v[begin:begin+512].to(device) for v in tensors[:6]]
            output.append(model(*batch)[0].cpu().numpy())
    return np.concatenate(output)


def select_scale(pred, data, cfg):
    baseline = metrics(data["old"], data)
    choices = []
    for scale in cfg["validation_scales"]:
        score = metrics(data["old"] + scale * (pred-data["old"]), data)
        if score["mae_physical"] <= cfg["mae_tolerance"] * baseline["mae_physical"]:
            choices.append((score["rmse_physical"], scale, score))
    if not choices:
        raise ValueError("Identity fallback must always be feasible")
    _, scale, score = min(choices, key=lambda item: (item[0], item[1]))
    return scale, score


def cache_hashes(out, station):
    return {name: sha(out / station / name) for name in (
        "train.npz", "validation.npz", "vocabulary.json", "prompts.json",
        "language_bank.npz", "random_bank.npz")}


def train(cfg, out):
    if (out / "test_lock.json").exists():
        raise ValueError("Test already unlocked; do not retrain this experiment version")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for station in cfg["stations"]:
        tr, va = load_cache(out, station, "train"), load_cache(out, station, "validation")
        hashes = cache_hashes(out, station)
        epsilon = max(1e-6, cfg["correction_sigma_fraction"] * float((tr["target"]-tr["old"]).std()))
        for arm in ARMS:
            xtr = inputs(tr, out, station, arm, split="train")
            xva = inputs(va, out, station, arm, split="validation")
            locks = []
            for lr in cfg["learning_rates"]:
                folder = out / station / arm / f"lr_{lr:g}"
                lock_path = folder / "lock.json"
                if lock_path.exists():
                    lock = read_json(lock_path)
                    if lock["cache_hashes"] != hashes or lock["checkpoint_sha256"] != sha(folder / "head.pt"):
                        raise ValueError("Trial artifacts changed")
                    locks.append(lock)
                    continue
                _seed_everything(2026)
                model = PhysicalLanguageCorrection(epsilon).to(device)
                optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg["weight_decay"])
                initial = predict(model, xva, device)
                np.testing.assert_array_equal(initial, va["old"])
                best = metrics(initial, va)["rmse_physical"]
                state, best_epoch, stale, history = copy.deepcopy(model.state_dict()), 0, 0, []
                for epoch in range(1, cfg["epochs"]+1):
                    model.train()
                    order = np.random.default_rng(2026+epoch).permutation(len(tr["old"]))
                    losses = []
                    for begin in range(0, len(order), cfg["batch"]):
                        ix = order[begin:begin+cfg["batch"]]
                        batch = [v[ix].to(device) for v in xtr]
                        optimizer.zero_grad(set_to_none=True)
                        pred, _, p, t = model(*batch[:6])
                        loss = F.mse_loss(pred, batch[7])
                        if arm in ALIGNED:
                            loss = loss + cfg["alignment_weight"] * state_alignment(p, t, batch[6], batch[5])
                        if not torch.isfinite(loss):
                            raise ValueError("Nonfinite loss; no silent trial exclusion")
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                        optimizer.step()
                        losses.append(loss.detach())
                    score = metrics(predict(model, xva, device), va)["rmse_physical"]
                    history.append(dict(epoch=epoch, validation_rmse=score,
                                        loss=float(torch.stack(losses).mean())))
                    print(f"[train] {station}/{arm}/lr{lr:g} epoch={epoch} val={score:.7f}", flush=True)
                    if score < best:
                        best, best_epoch, stale = score, epoch, 0
                        state = copy.deepcopy(model.state_dict())
                    else:
                        stale += 1
                    if stale >= cfg["patience"]:
                        break
                model.load_state_dict(state)
                prediction = predict(model, xva, device)
                scale, score = select_scale(prediction, va, cfg)
                folder.mkdir(parents=True, exist_ok=True)
                torch.save({"state_dict": {k: v.cpu() for k, v in state.items()}}, folder / "head.pt")
                lock = dict(station=station, arm=arm, seed=2026, lr=lr, epsilon=epsilon,
                    scale=scale, best_epoch=best_epoch, validation=score, history=history,
                    validation_raw=metrics(prediction, va),
                    parameters=sum(p.numel() for p in model.parameters()), cache_hashes=hashes,
                    checkpoint=str((folder / "head.pt").relative_to(ROOT)),
                    checkpoint_sha256=sha(folder / "head.pt"))
                write_json(lock_path, lock)
                locks.append(lock)
            selected = min(locks, key=lambda r: (r["validation"]["rmse_physical"], r["scale"], r["lr"]))
            write_json(out / station / arm / "selection.json", selected)


def require_all_locks(cfg, out):
    for station in cfg["stations"]:
        hashes = cache_hashes(out, station)
        for arm in ARMS:
            path = out / station / arm / "selection.json"
            if not path.exists():
                raise ValueError("Complete all 20 validation selections before ANY test access")
            lock = read_json(path)
            if lock["cache_hashes"] != hashes or sha(ROOT / lock["checkpoint"]) != lock["checkpoint_sha256"]:
                raise ValueError("Selected checkpoint/cache changed")


def test(cfg, formal, out):
    require_all_locks(cfg, out)
    selections = {f"{station}/{arm}": sha(out / station / arm / "selection.json")
                  for station in cfg["stations"] for arm in ARMS}
    lock_path = out / "test_lock.json"
    if lock_path.exists() and read_json(lock_path) != selections:
        raise ValueError("Validation selections changed after test unlock")
    write_json(lock_path, selections)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    expected = read_json(ROOT / "outputs/correction_tune_v1/test_confirmation_report.json")["per_station"]
    for station in cfg["stations"]:
        report_path = out / station / "test.json"
        if report_path.exists():
            continue
        data = frozen_features(station, "test", cfg, formal)
        old_score = metrics(data["old"], data)
        for key in ("rmse_physical", "mae_physical", "mbe_physical", "r2"):
            if abs(old_score[key]-expected[station]["FM_real"][key]) > 1e-6:
                raise ValueError(f"{station}: old formal test mismatch; do not silently replace baseline")
        results = {"old_psrc_chronos": old_score}
        arrays = {"starts": data["starts"], "target_std": data["target"],
                  "old_std": data["old"], "target_mu": data["mu"], "target_sd": data["sd"]}
        for arm in ARMS:
            lock = read_json(out / station / arm / "selection.json")
            model = PhysicalLanguageCorrection(lock["epsilon"]).to(device)
            model.load_state_dict(torch.load(ROOT / lock["checkpoint"], map_location=device,
                                             weights_only=True)["state_dict"], strict=True)
            raw = predict(model, inputs(data, out, station, arm, split="test"), device)
            pred = data["old"] + lock["scale"] * (raw-data["old"])
            results[arm] = dict(metrics=metrics(pred, data), raw_metrics=metrics(raw, data),
                                scale=lock["scale"], checkpoint_sha256=lock["checkpoint_sha256"])
            arrays[arm] = pred
        np.savez_compressed(out / station / "test_arrays.npz", **arrays)
        write_json(report_path, dict(status="exploratory_after_prior_test_exposure", results=results))


def main(action="plan", *, dry_run=False):
    cfg, formal, out = setup()
    print(f"Physical-language: 4 stations x {len(ARMS)} arms x 3 LRs = 60 fits; "
          "seed2026, L96/H16, 100ep/patience10. Frozen old PSRC+Chronos retained.")
    if dry_run or action == "plan":
        print(f"Output: {out}\nActions: audit -> prepare -> train; explicit test -> report")
        return
    if action not in ("audit", "prepare", "train", "test", "report"):
        raise ValueError(action)
    manifest(cfg, formal, out)
    if action == "audit":
        for station in cfg["stations"]:
            data = frozen_features(station, "validation", cfg, formal, audit_rows=64)
            write_json(out / station / "audit.json", dict(
                validation_rows=len(data["old"]), role_valid=data["role_valid"].tolist(),
                max_abs_reconstruction_error=float(data["old_max_abs_reconstruction_error"]),
                test_evaluated=False, trained=False))
            print(f"[audit OK] {station}: exact registered adapter located and reconstructed", flush=True)
    elif action == "prepare":
        prepare(cfg, formal, out)
    elif action == "train":
        train(cfg, out)
    elif action == "test":
        test(cfg, formal, out)
    else:
        report = {station: {"validation": {arm: read_json(out / station / arm / "selection.json")
                         for arm in ARMS}} for station in cfg["stations"]}
        for station in report:
            path = out / station / "test.json"
            if path.exists():
                report[station]["test"] = read_json(path)
        write_json(out / "comparison.json", report)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "audit", "prepare", "train", "test", "report"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    main(args.action, dry_run=args.dry_run)
