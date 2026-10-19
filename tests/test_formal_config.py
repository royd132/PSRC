import json

import pytest

from formal.config import DATASET_NAMES, MODEL_NAMES, load_experiment_config


CONFIG_PATH = "configs/formal_experiment.json"


def test_checked_in_config_is_the_locked_h16_matrix():
    config = load_experiment_config(CONFIG_PATH)

    assert tuple(dataset.name for dataset in config.datasets) == DATASET_NAMES
    assert config.models == MODEL_NAMES
    assert config.seeds == (2026,)
    assert config.horizons == (16,)
    assert config.seq_len == 96
    assert config.epochs == 100
    assert config.patience == 10
    assert len(config.jobs()) == 4 * len(MODEL_NAMES)


def test_locked_rnn_sensitivity_configs_are_explicit():
    raw = json.loads(open(CONFIG_PATH, encoding="utf-8").read())

    assert raw["baseline_overrides"]["lstm"] == {
        "lstm_d_model": 128,
        "lstm_layers": 3,
        "lstm_dropout": 0.3,
        "weight_decay": 0.0,
    }
    assert raw["baseline_overrides"]["gru"] == {
        "gru_d_model": 64,
        "gru_layers": 3,
        "gru_dropout": 0.3,
        "weight_decay": 0.0,
    }


def test_loader_rejects_nonformal_dimensions(tmp_path):
    raw = json.loads(open(CONFIG_PATH, encoding="utf-8").read())
    for field, value in (("horizons", [24]), ("seeds", [2027])):
        payload = dict(raw)
        payload[field] = value
        path = tmp_path / f"bad_{field}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError):
            load_experiment_config(path)
