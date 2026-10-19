"""Fixed joint loss-balance exploration for DKASC; single seed2026."""
from itertools import product
from pathlib import Path

from formal.semantic_explore_v2 import ROOT, main as run_campaign


def candidates(base):
    result = {}
    for factor, semantic, anchor, loss in product(
        (.7, 1.3), (.05, .2), (.25, 1.), ("mse", "huber")
    ):
        name = f"lr{factor:g}_sem{semantic:g}_anchor{anchor:g}_{loss}"
        result[name] = {**base, "lr": base["lr"] * factor,
                        "semantic_loss_weight": semantic,
                        "base_anchor_weight": anchor,
                        "loss_base": loss, "huber_delta": 1.}
    return result


def main(dry_run=False):
    run_campaign(dry_run=dry_run,
        source=ROOT / "outputs/semantic_explore_joint_v3",
        output=ROOT / "outputs/semantic_explore_dkasc_v4",
        stations=("dkasc_site31", "dkasc_site1a"),
        grid_factory=candidates, study="semantic_explore_dkasc_v4",
        extra_sources=(Path(__file__),))


if __name__ == "__main__":
    import sys
    main(dry_run="--dry-run" in sys.argv)
