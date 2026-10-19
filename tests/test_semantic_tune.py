from formal.semantic_tune import parameters, CANDIDATES, main, SEEDS


def test_candidate_multipliers_preserve_incumbent():
    base = {"lr": .001, "max_correction": .25, "semantic_loss_weight": .2,
            "base_anchor_weight": 1., "d_model": 96}
    assert parameters(base, "incumbent") == base
    for candidate in CANDIDATES:
        result = parameters(base, candidate)
        assert result["d_model"] == 96
        assert all(result[k] > 0 for k in base)
    assert parameters(base, "residual_double")["max_correction"] == .5
    assert base["max_correction"] == .25


def test_plan_does_not_train():
    assert SEEDS == (2026,)
    main(dry_run=True)
