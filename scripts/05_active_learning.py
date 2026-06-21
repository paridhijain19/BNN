"""Stage 05: run active learning and benchmark acquisition strategies vs. random."""

from __future__ import annotations

import argparse

from _common import load_config
from bbnjax.active_learning import compare_acquisitions
from bbnjax.tracking import ExperimentTracker
from bbnjax.viz import plots, set_style


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/active_learning.yaml")
    parser.add_argument(
        "--acquisitions", nargs="+", default=["max_variance", "bald", "random"]
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_style()

    out_dir = cfg["active_learning"].get("out_dir", "artifacts/active_learning")
    tracker = ExperimentTracker(out_dir, config=dict(cfg), seed=int(cfg.get("seed", 0)))

    print(f"[AL] comparing acquisitions: {args.acquisitions}")
    histories = compare_acquisitions(cfg, acquisitions=tuple(args.acquisitions))

    summary = {name: h.as_dict() for name, h in histories.items()}
    tracker.log_summary(summary)
    plots.plot_active_learning(
        histories, metric="test_rmse_mean", save=tracker.path("al_rmse.png")
    )
    plots.plot_active_learning(
        histories, metric="test_nll", save=tracker.path("al_nll.png")
    )
    print(f"[done] active-learning results written to {out_dir}")


if __name__ == "__main__":
    main()
