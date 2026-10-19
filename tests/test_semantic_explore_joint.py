from formal.semantic_explore_joint import candidates


def test_joint_grid_is_bounded_and_preserves_structure():
    base = dict(lr=.001, batch=32, loss_base="mse", huber_delta=1.,
                d_model=96, heads=4, semantic_loss_weight=.2)
    grid = candidates(base)
    assert len(grid) == 12
    assert {p["lr"] for p in grid.values()} == {.0005, .001, .002}
    assert {p["batch"] for p in grid.values()} == {64, 128}
    assert {p["loss_base"] for p in grid.values()} == {"mse", "huber"}
    assert all(p["d_model"] == 96 and p["semantic_loss_weight"] == .2 for p in grid.values())
    assert base["batch"] == 32
