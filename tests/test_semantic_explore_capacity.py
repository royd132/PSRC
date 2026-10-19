import json

from formal.semantic_explore_capacity import candidates


def test_capacity_grid_changes_only_registered_dimensions():
    base = dict(d_model=96, gtr_period=12, patch_hours=[1, 2, 4, 8],
                lr=.001, heads=4, token_statistics="physical", batch=64,
                semantic_loss_weight=.3, max_gate=.5, max_correction=.25)
    original = json.dumps(base, sort_keys=True)
    grid = candidates(base)
    assert len(grid) == 36
    assert len({json.dumps(p, sort_keys=True) for p in grid.values()}) == 36
    assert candidates(base) == grid
    assert json.dumps(base, sort_keys=True) == original
    mutable = {"d_model", "gtr_period", "patch_hours", "lr"}
    for p in grid.values():
        assert {k: v for k, v in p.items() if k not in mutable} == {
            k: v for k, v in base.items() if k not in mutable}
        assert p["d_model"] % p["heads"] == 0
        assert 96 % p["gtr_period"] == 0
        assert all(1 <= round(h / .25) <= 96 for h in p["patch_hours"])
    assert {p["d_model"] for p in grid.values()} == {64, 128, 192}
    assert {p["gtr_period"] for p in grid.values()} == {12, 24, 48}
