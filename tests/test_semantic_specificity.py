"""Small synthetic checks; no dataset training or test evaluation."""
import torch
from types import SimpleNamespace
import json

from formal.semantic_study import ARMS, SEEDS, LR_FACTORS, ablation
from layers.physical_semantic import PhysicalSemanticEncoder, compute_generic_tokens
from layers.residual_corrector import SemanticResidualCalibration


def encoder(**kwargs):
    return PhysicalSemanticEncoder({}, [0.] * 4, [1.] * 4, 0., 1.,
                                   torch.zeros(6, 6), torch.ones(6, 6), **kwargs)


def test_regroup_preserves_values_and_parameters():
    x = torch.randn(3, 96, 4)
    original, regroup = encoder(), encoder(regroup_seed=4101)
    a, b = original.standardized_tokens(x), regroup.standardized_tokens(x)
    assert torch.equal(a.flatten(1).sort(1).values, b.flatten(1).sort(1).values)
    assert not torch.equal(a, b)
    assert list(original.state_dict()) == list(regroup.state_dict())
    assert sum(p.numel() for p in original.parameters()) == sum(p.numel() for p in regroup.parameters())
    assert torch.equal(b, regroup.standardized_tokens(x))


def test_generic_statistics_and_shared_retrieval():
    x = torch.ones(3, 96, 4)
    tokens = compute_generic_tokens(x)
    assert tokens.shape == (3, 6, 6)
    assert torch.equal(tokens[..., 0], torch.ones(3, 6))
    assert torch.equal(tokens[..., 1], torch.zeros(3, 6))
    assert torch.allclose(tokens[..., 5], torch.zeros(3, 6), atol=1e-6)
    head = SemanticResidualCalibration(32, 32, 16, shared_retrieval=True)
    result = head(torch.randn(3, 4, 32), torch.randn(3, 32),
                  torch.randn(3, 16), torch.randn(3, 6, 32))
    assert result[0].shape == (3, 16)
    assert torch.isfinite(result[0]).all()


def test_plan_budget():
    assert len(ARMS) * len(SEEDS) * len(LR_FACTORS) * 4 == 480
    assert ablation("physical") == set()
    assert ablation("concat_stats") == {"concat_stats"}


def test_separate_training_and_locked_test(tmp_path, monkeypatch):
    import formal.semantic_study as study
    config = SimpleNamespace(datasets=[SimpleNamespace(name="station")],
                             train_fraction=.7, validation_end_fraction=.85,
                             psrc={"gate": True}, dataset=lambda _: SimpleNamespace(path="unused"))
    bundle = SimpleNamespace(validation=SimpleNamespace(starts="validation"),
                             test=SimpleNamespace(starts="test"))
    monkeypatch.setattr(study, "OUT", tmp_path)
    monkeypatch.setattr(study, "load_experiment_config", lambda _: config)
    monkeypatch.setattr(study, "protocol", lambda _: {"test": "immutable"})
    monkeypatch.setattr(study, "load_dataset", lambda *args: bundle)
    monkeypatch.setattr(study, "_parameter_record", lambda *args: ("fixed", {"lr": .001}))
    monkeypatch.setattr(study, "psrc_settings", lambda *args: {})
    monkeypatch.setattr(study, "_train_model", lambda *args, **kw:
                        (torch.nn.Linear(1, 1), [], SimpleNamespace(best_epoch=1), 128))
    monkeypatch.setattr(study, "_loader", lambda bundle, starts, *args: starts)
    evaluated = []

    def evaluate(model, loader, *args, collect=False):
        evaluated.append(loader)
        metrics = {"mse_standardized": 1., "rmse_physical": 1.,
                   "mae_physical": .5, "mbe_physical": 0., "r2": .9}
        return (metrics, {}) if collect else metrics

    monkeypatch.setattr(study, "_evaluate", evaluate)
    monkeypatch.setattr(study, "_model_config", lambda *args, **kwargs: None)
    monkeypatch.setattr(study, "build_model", lambda *args: torch.nn.Linear(1, 1))
    monkeypatch.setattr(study, "save_prediction_artifacts", lambda *args: None)
    study.main("train")
    assert evaluated == ["validation"] * 120
    selection = json.loads((tmp_path / "station/physical/selection.json").read_text())
    assert selection["lr_factor"] == .5
    study.main("test")
    assert evaluated.count("test") == 40
    study.main("test")
    assert evaluated.count("test") == 40  # Completed test passes are never repeated.
    assert (tmp_path / "summary.csv").is_file()
