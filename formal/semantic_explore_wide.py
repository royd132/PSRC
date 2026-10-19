"""Wider fixed optimization search for DKASC, without token/topology edits."""
import random
from itertools import product
from pathlib import Path

from formal.semantic_explore_v2 import ROOT, main as run_campaign


def candidates(base):
    # Balanced marginal coverage, not the full Cartesian product. The private
    # RNG fixes the candidate list; every training run still uses seed 2026.
    levels = {
        "lr": (5e-5, 1e-4, 3e-4, 1e-3, 3e-3, 5e-3),
        "weight_decay": (0., 1e-6, 1e-4, 1e-2),
        "batch": (32, 64, 128, 256),
        "dropout": (0., .05, .15, .3),
        "optimizer": ("adam", "adamw"),
        "scheduler": ("cosine", "cosine_warmup", "none"),
        "sampling_policy": ("epoch_uniform", "uniform"),
        "max_gate": (.1, .25, .5, 1.),
        "max_correction": (.1, .25, .5, 1., 1.5, 2.),
        "semantic_loss_weight": (.01, .05, .1, .3, .6, 1.),
        "base_anchor_weight": (0., .25, 1., 4.),
        "residual_alpha": (0., 1., 5., 10.),
        "daylight_weight": (0., .25, .5, 1.),
        "ramp_weight": (0., .05, .1, .3),
        "loss_base": ("mse", "huber"),
        "huber_delta": (.1, .25, .5, 1., 2., 4.),
    }
    rng = random.Random(2026)
    columns = {}
    for key, values in levels.items():
        column = list(values) * (24 // len(values))
        rng.shuffle(column)
        columns[key] = column
    result = {}
    for i, (lr, wd) in enumerate(product(levels["lr"], levels["weight_decay"])):
        result[f"optimizer_{i:02d}"] = {**base, "lr": lr, "weight_decay": wd}
        params = {**base, **{key: values[i] for key, values in columns.items()}}
        if params["loss_base"] == "mse":
            params["huber_delta"] = 1.  # Inactive in MSE; avoid misleading values.
        result[f"joint_{i:02d}"] = params
    return result


def main(dry_run=False):
    run_campaign(
        dry_run=dry_run,
        source=ROOT / "outputs/semantic_explore_dkasc_v4",
        output=ROOT / "outputs/semantic_explore_wide_v5",
        stations=("dkasc_site31", "dkasc_site1a"),
        grid_factory=candidates,
        study="semantic_explore_wide_v5",
        extra_sources=(Path(__file__),),
    )


if __name__ == "__main__":
    import sys
    main(dry_run="--dry-run" in sys.argv)
