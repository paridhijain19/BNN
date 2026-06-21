"""Stage 04: calibrate emulator predictive uncertainties and check reliability.

Trains an NN+GP, fits a per-target variance calibrator on the validation split,
and compares reliability before vs. after calibration on the test split.
"""

from __future__ import annotations

import argparse

from _common import load_config, load_or_generate_bundle, train_core
from bbnjax.models import NNGPEmulator
from bbnjax.tracking import ExperimentTracker
from bbnjax.uq import metrics
from bbnjax.uq.calibration import VarianceCalibrator
from bbnjax.viz import plots, set_style


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train_mlp.yaml")
    parser.add_argument("--out", default="artifacts/calibration")
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_style()

    bundle = load_or_generate_bundle(cfg)
    tracker = ExperimentTracker(args.out, config=dict(cfg), seed=int(cfg.get("seed", 0)))

    nn, _ = train_core("mlp", bundle, cfg["model"], cfg["train"], seed=int(cfg.get("seed", 0)))
    X, T = bundle.xy("train")
    emulator = NNGPEmulator.fit(nn, X, T, gp_steps=200)

    X_val, T_val = bundle.xy("val")
    X_test, T_test = bundle.xy("test")
    mval, vval = emulator.predict_batch(X_val)
    calib = VarianceCalibrator.fit(mval, vval, T_val)
    print("[calib] per-target std scale factors:", [float(s) for s in calib.scale])

    mean, var = emulator.predict_batch(X_test)
    var_cal = calib.apply(var)

    before = metrics.calibration_error(mean, var, T_test)
    after = metrics.calibration_error(mean, var_cal, T_test)
    summary = {
        "calibration_error_before": before,
        "calibration_error_after": after,
        "scale": [float(s) for s in calib.scale],
        "picp68_before": [float(v) for v in metrics.picp(mean, var, T_test, 0.68)],
        "picp68_after": [float(v) for v in metrics.picp(mean, var_cal, T_test, 0.68)],
    }
    print("[calib] summary:", summary)
    tracker.log_summary(summary)

    e0, o0 = metrics.reliability_curve(mean, var, T_test)
    e1, o1 = metrics.reliability_curve(mean, var_cal, T_test)
    fig = plots.plot_reliability(e0, o0, label="before")
    fig.axes[0].plot(e1, o1, "s-", label="after", color="C1")
    fig.axes[0].legend()
    fig.savefig(tracker.path("reliability_calibrated.png"))
    print(f"[done] artifacts written to {args.out}")


if __name__ == "__main__":
    main()
