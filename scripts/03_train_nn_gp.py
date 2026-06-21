"""Stage 03: fit the NN + GP residual emulator and evaluate its uncertainties."""

from __future__ import annotations

import argparse

from _common import load_config, load_or_generate_bundle, train_core
from bbnjax.models import NNGPEmulator, make_physical
from bbnjax.tracking import ExperimentTracker, save_emulator
from bbnjax.uq import metrics
from bbnjax.viz import plots, set_style


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train_mlp.yaml")
    parser.add_argument("--out", default="artifacts/nn_gp")
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_style()

    bundle = load_or_generate_bundle(cfg)
    tracker = ExperimentTracker(args.out, config=dict(cfg), seed=int(cfg.get("seed", 0)))

    # Train the NN mean, then fit a GP to its residuals.
    nn, _ = train_core("mlp", bundle, cfg["model"], cfg["train"], seed=int(cfg.get("seed", 0)))
    X, T = bundle.xy("train")
    emulator = NNGPEmulator.fit(nn, X, T, kernel="matern52", gp_steps=200)

    X_test, T_test = bundle.xy("test")
    mean, var = emulator.predict_batch(X_test)
    summ = metrics.summary(mean, T_test, var, target_names=list(bundle.target_names))
    print("[eval] NN+GP metrics:", summ)
    tracker.log_summary(summ)

    save_emulator(tracker.path("emulator.pkl"), make_physical(emulator, bundle), bundle.space)
    z = metrics.standardized_residuals(mean, var, T_test)
    plots.plot_standardized_residuals(z, bundle.target_names, save=tracker.path("residuals.png"))
    expected, observed = metrics.reliability_curve(mean, var, T_test)
    plots.plot_reliability(expected, observed, save=tracker.path("reliability.png"), label="NN+GP")
    print(f"[done] artifacts written to {args.out}")


if __name__ == "__main__":
    main()
