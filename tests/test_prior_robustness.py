"""Target-free pairing and fixed-energy train augmentation contracts."""
from types import SimpleNamespace

import numpy as np

from formal.prior_robustness import corrupt_prior, features, matched_indices


def test_reconstruction_restores_historical_parameter_flags():
    import torch
    from formal.prior_robustness import restore_evaluation_flags
    model = torch.nn.Module()
    model.backbone = torch.nn.Linear(2, 2)
    model.pc_fra_adapter = torch.nn.Linear(2, 2)
    original = {k: v.clone() for k, v in model.state_dict().items()}
    restore_evaluation_flags(model, "validation")
    assert not model.backbone.weight.requires_grad
    assert model.pc_fra_adapter.weight.requires_grad
    restore_evaluation_flags(model, "test")
    assert all(p.requires_grad for p in model.parameters())
    for k, v in model.state_dict().items():
        torch.testing.assert_close(v, original[k], rtol=0, atol=0)


def test_matched_donors_preserve_strata_and_exclude_overlapping_windows():
    starts = np.arange(12)*120
    values = np.zeros((12, 6))
    values[:, 0] = .25
    values[:, 1] = .1
    values[:, 3] = 3
    values[:, 5] = 1
    values[6:, 4] = 2
    for seed in (4101, 4102):
        donor, matched = matched_indices(starts, values, [.2, .4], seed)
        assert matched.all()
        assert (np.abs(starts[donor]-starts) >= 112).all()
        np.testing.assert_array_equal(values[donor], values)
    donor, matched = matched_indices(np.arange(5), values[:5], [.2, .4], 1)
    assert not matched.any()
    np.testing.assert_array_equal(donor, np.arange(5))


def test_augmentation_has_same_mask_energy_and_preserves_clean_rows():
    prior = np.full((100, 16), 2., dtype=np.float32)
    values = np.zeros((100, 6))
    values[:, 3] = np.arange(100) % 4
    noise, selected = corrupt_prior(prior, values, 5., "noise")
    shaped, selected2 = corrupt_prior(prior, values, 5., "state")
    np.testing.assert_array_equal(selected, selected2)
    np.testing.assert_array_equal(noise[~selected], prior[~selected])
    np.testing.assert_array_equal(shaped[~selected], prior[~selected])
    np.testing.assert_allclose(np.mean((noise-prior)**2, axis=1),
                               np.mean((shaped-prior)**2, axis=1), atol=1e-7)
    np.testing.assert_array_equal(corrupt_prior(prior, values, 5., "clean")[0], prior)
    assert not np.array_equal(noise, shaped)


def test_operating_features_do_not_read_future_targets():
    bundle = SimpleNamespace(raw=np.ones((150, 2), dtype=np.float32), capacity=5.,
        timestamps=np.datetime64("2020-01-01") + np.arange(150)*np.timedelta64(15, "m"),
        daylight_index=None)
    before = features(bundle, [0])
    bundle.raw[96:, -1] = 99999
    np.testing.assert_array_equal(features(bundle, [0]), before)


def test_training_hook_leaves_validation_pack_unchanged(monkeypatch, tmp_path):
    from contextlib import nullcontext
    from formal import prior_robustness as study
    from layers.pc_fra import pack_width

    train_starts, val_starts = np.arange(100), np.arange(200, 300)
    bundle = SimpleNamespace(train=SimpleNamespace(starts=train_starts),
        validation=SimpleNamespace(starts=val_starts), capacity=5.)
    original = np.full((100, pack_width(16)), 2., dtype=np.float32)
    monkeypatch.setattr(study.ev, "prior_hooks", lambda *args: nullcontext())
    monkeypatch.setattr(study.fm, "build_pack", lambda *args: SimpleNamespace(embeddings=original.copy()))
    monkeypatch.setattr(study.fm, "contract_prior", lambda x, *args: np.clip(x, 0, 5))
    monkeypatch.setattr(study, "features", lambda *args: np.zeros((100, 6)))
    monkeypatch.setattr(study, "OUT", tmp_path)
    with study.training_hooks("station", "state", None):
        assert study.fm.PATIENCE == 101
        train = study.fm.build_pack(bundle, train_starts, 96, 16, {}, {}, train_starts)
        val = study.fm.build_pack(bundle, val_starts, 96, 16, {}, {}, val_starts)
    assert not np.array_equal(train.embeddings, original)
    np.testing.assert_array_equal(val.embeddings, original)
