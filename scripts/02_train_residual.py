"""Stage 02: train the residual emulator (frozen linear baseline + NN correction)."""

from __future__ import annotations

import argparse

import jax

from _common import load_config, load_or_generate_bundle, train_core
from bbnjax.models import make_physical
from bbnjax.tracking import ExperimentTracker, save_emulator
from bbnjax.uq import metrics
from bbnjax.viz import plots, set_style


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train_residual.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_style()

    bundle = load_or_generate_bundle(cfg)
    out_dir = cfg.get("out_dir", "artifacts/residual")
    tracker = ExperimentTracker(out_dir, config=dict(cfg), seed=int(cfg.get("seed", 0)))

    core, result = train_core(
        "residual", bundle, cfg["model"], cfg["train"], seed=int(cfg.get("seed", 0))
    )

    X_test, T_test = bundle.xy("test")
    pred = jax.vmap(core)(X_test)
    summ = metrics.summary(pred, T_test, target_names=list(bundle.target_names))
    print("[eval] residual emulator test metrics:", summ)

    tracker.log_summary(summ)
    save_emulator(tracker.path("emulator.pkl"), make_physical(core, bundle), bundle.space)
    plots.plot_training_curves(
        result.train_loss, result.val_loss, save=tracker.path("training_curve.png")
    )
    plots.plot_pred_vs_true(pred, T_test, bundle.target_names, save=tracker.path("pred_vs_true.png"))
    print(f"[done] artifacts written to {out_dir}")


if __name__ == "__main__":
    main()
