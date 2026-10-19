"""Fixed joint optimization grid; seed2026, exploratory, no architecture changes."""
from pathlib import Path
from formal.semantic_explore_v2 import ROOT, main as run_campaign


def candidates(base):
    result = {}
    for factor in (.5, 1., 2.):
        for batch in (64, 128):
            for loss in ("mse", "huber"):
                params = {**base, "lr": base["lr"] * factor, "batch": batch,
                          "loss_base": loss, "huber_delta": .5}
                # Keep the baseline's feature/loss weights and architecture.
                if params == base:
                    continue
                result[f"lr{factor:g}_batch{batch}_{loss}"] = params
    return result


def main(dry_run=False):
    run_campaign(dry_run=dry_run,
        source=ROOT / "outputs/semantic_explore_physical_v2",
        output=ROOT / "outputs/semantic_explore_joint_v3",
        stations=("dkasc_site31", "dkasc_site1a", "pvod_station02"),
        grid_factory=candidates, study="semantic_explore_joint_v3",
        extra_sources=(Path(__file__),))


if __name__ == "__main__":
    import sys
    main(dry_run="--dry-run" in sys.argv)
