"""DKASC capacity/time-scale search; unchanged token definitions and routing.

Exploratory forecast tuning, not a parameter-matched semantic-specificity test.
"""
from itertools import product
from pathlib import Path

from formal.semantic_explore_v2 import ROOT, main as run_campaign


def candidates(base):
    result = {}
    for width, period, patches, lr in product(
        (128, 192, 64), (12, 24, 48),
        ((1., 2., 4., 8.), (2., 4., 8., 12.)), (3e-4, 1.5e-3),
    ):
        scale = "standard" if patches[0] == 1. else "long"
        name = f"d{width}_gtr{period}_{scale}_lr{lr:g}"
        result[name] = {**base, "d_model": width, "gtr_period": period,
                        "patch_hours": list(patches), "lr": lr}
    return result


def main(dry_run=False):
    run_campaign(
        dry_run=dry_run,
        source=ROOT / "outputs/semantic_explore_wide_v5",
        output=ROOT / "outputs/semantic_explore_capacity_v6",
        stations=("dkasc_site31", "dkasc_site1a"),
        grid_factory=candidates,
        study="semantic_explore_capacity_v6_not_parameter_matched",
        extra_sources=(Path(__file__),),
    )


if __name__ == "__main__":
    import sys
    main(dry_run="--dry-run" in sys.argv)
