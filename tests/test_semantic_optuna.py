import optuna
from formal.semantic_optuna import suggest, ARMS, TRIALS, main


def test_space_and_budget():
    values = {"lr": .001, "weight_decay": 1e-5, "dropout": .1,
              "max_correction": .25, "max_gate": .5, "semantic_loss_weight": .2,
              "base_anchor_weight": 1., "residual_alpha": 1.}
    base = {**values, "d_model": 96}
    output = suggest(optuna.trial.FixedTrial(values), base)
    assert output == base
    assert len(ARMS) * TRIALS * 4 == 240
    main(dry_run=True)
