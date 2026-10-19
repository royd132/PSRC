from formal.semantic_explore_dkasc import candidates


def test_fixed_grid_keeps_architecture():
    base = dict(lr=.001, d_model=96, batch=64, max_gate=.5)
    grid = candidates(base)
    assert len(grid) == 16
    assert len({tuple(sorted(p.items())) for p in grid.values()}) == 16
    assert all(p["d_model"] == 96 and p["batch"] == 64 and p["max_gate"] == .5
               for p in grid.values())
    assert {p["semantic_loss_weight"] for p in grid.values()} == {.05, .2}
    assert {p["base_anchor_weight"] for p in grid.values()} == {.25, 1.}
    assert base == dict(lr=.001, d_model=96, batch=64, max_gate=.5)
