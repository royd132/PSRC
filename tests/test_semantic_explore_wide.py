import json

from formal.semantic_explore_wide import candidates


def test_wide_grid_reproducible_and_keeps_token_architecture():
    base = dict(lr=.001, weight_decay=1e-7, d_model=96, heads=4,
                batch=64, patch_hours=[1, 2, 4, 8], gtr_period=12,
                token_statistics="physical", loss_base="mse")
    original = json.dumps(base, sort_keys=True)
    grid = candidates(base)
    assert len(grid) == 48
    assert grid == candidates(base)
    assert len({json.dumps(p, sort_keys=True) for p in grid.values()}) == 48
    assert json.dumps(base, sort_keys=True) == original
    for p in grid.values():
        for key in ("d_model", "heads", "patch_hours", "gtr_period", "token_statistics"):
            assert p[key] == base[key]
        assert 5e-5 <= p["lr"] <= 5e-3
        assert 0. <= p["weight_decay"] <= .01
    joint = [p for name, p in grid.items() if name.startswith("joint_")]
    assert {p["batch"] for p in joint} == {32, 64, 128, 256}
    assert {p["max_gate"] for p in joint} == {.1, .25, .5, 1.}
    assert {p["scheduler"] for p in joint} == {"cosine", "cosine_warmup", "none"}
    assert {p["base_anchor_weight"] for p in joint} == {0., .25, 1., 4.}
