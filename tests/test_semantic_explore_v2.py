from formal.semantic_explore_v2 import candidates


def test_fixed_grid_preserves_architecture_and_base():
    base = dict(batch=64, optimizer="adam", d_model=96, heads=4, loss_base="mse")
    grid = candidates(base)
    assert len(grid) == 8
    assert grid["batch_half"]["batch"] == 32
    assert grid["batch_double"]["batch"] == 128
    assert grid["optimizer_switch"]["optimizer"] == "adamw"
    assert grid["huber_0.25"]["huber_delta"] == .25
    assert all(p["d_model"] == 96 and p["heads"] == 4 for p in grid.values())
    assert base["batch"] == 64 and base["loss_base"] == "mse"
