import json

import numpy as np
import pytest
import torch

from formal import physical_language as study
from formal.physical_prompts import (fit_vocabulary, last_valid_hidden,
                                     numeric_features, prompt_bank, state_codes)
from models.physical_language import PhysicalLanguageCorrection, state_alignment


def sample_inputs(n=8):
    generator = torch.Generator().manual_seed(2026)
    return [torch.randn(n, d, generator=generator) for d in (16, 16, 16)] + [
        torch.randn(n, 6, 32, generator=generator),
        torch.randn(n, 6, 768, generator=generator), torch.ones(n, 6)]


def test_zero_initialization_is_exact_and_output_can_learn():
    model = PhysicalLanguageCorrection(.1)
    x = sample_inputs()
    pred, delta, _, _ = model(*x)
    assert torch.equal(pred, x[0])
    assert torch.count_nonzero(delta) == 0
    ((pred - (x[0] + 1)) ** 2).mean().backward()
    assert model.output.weight.grad.abs().sum() > 0
    with torch.no_grad():
        model.output.weight.normal_()
    assert model(*x)[1].abs().max() <= .1


def test_missing_category_does_not_affect_prediction():
    model = PhysicalLanguageCorrection(.1)
    with torch.no_grad():
        model.output.weight.normal_()
    x = sample_inputs()
    x[5][:, 3] = 0
    before = model(*x)[0]
    x[3][:, 3] += 100
    x[4][:, 3] -= 100
    torch.testing.assert_close(model(*x)[0], before, rtol=0, atol=0)


def test_alignment_prefers_correct_states_and_has_finite_gradient():
    p = torch.eye(4).view(4, 1, 4).requires_grad_()
    codes = torch.arange(4).view(4, 1)
    mask = torch.ones(4, 1)
    good = state_alignment(p, p.detach(), codes, mask)
    bad = state_alignment(p, p.detach().roll(1, 0), codes, mask)
    assert good < bad
    bad.backward()
    assert torch.isfinite(p.grad).all()


@pytest.mark.parametrize("missing", [True, False])
def test_no_false_negatives_for_identical_states_or_missing_category(missing):
    p = torch.randn(4, 6, 32, requires_grad=True)
    codes = torch.zeros(4, 6, dtype=torch.long)
    available = torch.zeros(4, 6) if missing else torch.ones(4, 6)
    loss = state_alignment(p, torch.randn_like(p), codes, available)
    assert loss.item() == 0
    loss.backward()
    assert torch.isfinite(p.grad).all()


def vocabulary_example():
    raw = np.random.default_rng(2026).normal(size=(50, 6, 6))
    valid = np.array([1, 0, 1, 0, 1, 1], dtype=bool)
    return raw, fit_vocabulary(raw, valid)


def test_vocabulary_is_train_only_with_explicit_missing_categories():
    raw, vocabulary = vocabulary_example()
    before = json.dumps(vocabulary, sort_keys=True)
    codes = state_codes(raw * 1000, vocabulary)
    assert json.dumps(vocabulary, sort_keys=True) == before
    active = np.array(vocabulary["active"])
    assert not active[[1, 3]].any()
    assert not active[:, 5].any()
    assert not active[2:, 3:].any()
    assert np.all(codes[:, [1, 3]] == 0)
    prompts, keys = prompt_bank(vocabulary)
    lookup = dict(zip(map(tuple, keys), prompts))
    assert "unavailable" in lookup[(1, 0)]
    assert "unavailable" in lookup[(3, 0)]
    for row in codes:
        assert all((k, int(code)) in lookup for k, code in enumerate(row))


def test_numerical_control_is_dense_reproducible_and_finite():
    raw, vocabulary = vocabulary_example()
    features = numeric_features(raw, vocabulary)
    assert features.shape == (50, 6, 768)
    assert np.isfinite(features).all()
    np.testing.assert_array_equal(features, numeric_features(raw, vocabulary))
    assert features[:, 0].std(0).mean() > .01


def test_last_valid_token_handles_both_padding_sides():
    hidden = torch.arange(24).reshape(2, 4, 3)
    mask = torch.tensor([[1, 1, 0, 0], [0, 0, 1, 1]])
    actual = last_valid_hidden(hidden, mask)
    assert torch.equal(actual, torch.stack([hidden[0, 1], hidden[1, 3]]))
    with pytest.raises(ValueError, match="empty"):
        last_valid_hidden(hidden, torch.zeros_like(mask))


def test_shuffled_control_never_uses_validation_or_test_donors():
    train_codes = np.arange(60).reshape(10, 6)
    permuted = study.shuffled_codes(train_codes, train_codes, "train")
    assert all(not np.array_equal(a, b) for a, b in zip(train_codes, permuted))
    assert set(map(tuple, permuted)) == set(map(tuple, train_codes))
    for split in ("validation", "test"):
        donors = study.shuffled_codes(np.full((20, 6), 999), train_codes, split)
        assert set(map(tuple, donors)) <= set(map(tuple, train_codes))


def test_validation_scale_can_reject_harmful_branch():
    target = np.array([[1., 2.], [3., 4.]])
    data = dict(old=target + .1, target=target, mu=0., sd=1., capacity=5.)
    cfg = dict(validation_scales=[0., .25, .5, 1.], mae_tolerance=1.005)
    scale, _ = study.select_scale(data["old"] + 1, data, cfg)
    assert scale == 0
    scale, score = study.select_scale(target, data, cfg)
    assert scale == 1 and score["rmse_physical"] == 0


def test_nonfinite_metrics_fail_instead_of_dropping_windows():
    data = dict(target=np.ones((2, 2)), mu=0., sd=1., capacity=5.)
    with pytest.raises(ValueError, match="Nonfinite"):
        study.metrics(np.full((2, 2), np.nan), data)


def test_test_access_blocked_before_all_validation_selections(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "cache_hashes", lambda *args: {})
    monkeypatch.setattr(study, "frozen_features", lambda *args, **kwargs: pytest.fail("test read"))
    with pytest.raises(ValueError, match="20 validation"):
        study.test({"stations": {"a": {}}}, None, tmp_path)


def test_no_retraining_after_test_unlock(tmp_path):
    (tmp_path / "test_lock.json").touch()
    with pytest.raises(ValueError, match="already unlocked"):
        study.train({}, tmp_path)


def test_single_entry_dry_run_never_loads_checkpoints(monkeypatch):
    from formal.runner import main
    monkeypatch.setattr(study, "manifest", lambda *args: pytest.fail("artifact access"))
    monkeypatch.setattr(study, "train", lambda *args: pytest.fail("training"))
    main(["--physical-language", "train", "--dry-run"])
    with pytest.raises(ValueError, match="alone"):
        main(["--physical-language", "train", "--semantic-tune"])


def test_protocol_is_single_seed_and_equal_budget():
    cfg, _, _ = study.setup()
    assert cfg["seed"] == 2026
    assert len(cfg["stations"]) * len(study.ARMS) * len(cfg["learning_rates"]) == 60
    assert cfg["epochs"] == 100 and cfg["patience"] == 10
    counts = [sum(p.numel() for p in PhysicalLanguageCorrection(.1).parameters()) for _ in study.ARMS]
    assert len(set(counts)) == 1


def test_synthetic_train_lock_test_report_pipeline(tmp_path, monkeypatch):
    """Tiny generated data only: no station data, old checkpoints or GPT-2."""
    monkeypatch.setattr(study, "ROOT", tmp_path)
    cfg = dict(stations={"toy": {}}, correction_sigma_fraction=.25,
               learning_rates=[.001], weight_decay=.001, epochs=2, patience=1,
               batch=4, alignment_weight=.01, validation_scales=[0., 1.], mae_tolerance=1.005)
    out = tmp_path / "outputs/toy"
    folder = out / "toy"
    folder.mkdir(parents=True)
    rng = np.random.default_rng(2026)
    target = rng.normal(size=(8, 16)).astype(np.float32)
    data = dict(old=target + .1, target=target, prior=target + .2,
                disagreement=np.full_like(target, .1),
                raw=rng.normal(size=(8, 6, 6)).astype(np.float32),
                physical=rng.normal(size=(8, 6, 32)).astype(np.float32),
                role_valid=np.ones(6, dtype=bool), mu=2., sd=1., capacity=10.,
                starts=np.arange(8))
    for split in ("train", "validation"):
        np.savez_compressed(folder / f"{split}.npz", **data)
    vocabulary = fit_vocabulary(data["raw"], data["role_valid"])
    study.write_json(folder / "vocabulary.json", vocabulary)
    prompts, keys = prompt_bank(vocabulary)
    study.write_json(folder / "prompts.json", dict(prompts=prompts, keys=keys.tolist()))
    for name in ("language_bank", "random_bank"):
        np.savez_compressed(folder / f"{name}.npz", keys=keys,
                            embeddings=rng.normal(size=(len(keys), 768)).astype(np.float32))
    study.train(cfg, out)
    study.require_all_locks(cfg, out)
    counts = set()
    for arm in study.ARMS:
        lock = study.read_json(folder / arm / "selection.json")
        counts.add(lock["parameters"])
        assert lock["seed"] == 2026 and lock["best_epoch"] <= 2
    assert counts == {43670}
    study.write_json(tmp_path / "outputs/correction_tune_v1/test_confirmation_report.json",
                     {"per_station": {"toy": {"FM_real": study.metrics(data["old"], data)}}})
    calls = []
    def frozen(*args, **kwargs):
        calls.append(args[1])
        return data
    monkeypatch.setattr(study, "frozen_features", frozen)
    study.test(cfg, None, out)
    assert calls == ["test"]
    assert set(study.read_json(folder / "test.json")["results"]) == {"old_psrc_chronos", *study.ARMS}
    study.test(cfg, None, out)
    assert calls == ["test"]  # Resume never reevaluates an already recorded station.
    lock = study.read_json(folder / "numeric/selection.json")
    lock["scale"] = .75
    study.write_json(folder / "numeric/selection.json", lock)
    with pytest.raises(ValueError, match="changed after test"):
        study.test(cfg, None, out)
