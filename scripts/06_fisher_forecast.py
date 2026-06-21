"""Stage 06: autodiff sensitivities + Fisher forecast through the emulator."""

from __future__ import annotations

import argparse

import numpy as np

from _common import load_config, load_or_generate_bundle, train_core
from bbnjax.inference.fisher import ObservationalData, forecast_report, parameter_covariance, fisher_matrix
from bbnjax.inference.sensitivity import abundance_jacobian, finite_difference_jacobian, log_sensitivity
from bbnjax.models import make_physical
from bbnjax.tracking import ExperimentTracker
from bbnjax.viz import plots, set_style


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train_mlp.yaml")
    parser.add_argument("--out", default="artifacts/fisher")
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_style()

    bundle = load_or_generate_bundle(cfg)
    tracker = ExperimentTracker(args.out, config=dict(cfg), seed=int(cfg.get("seed", 0)))

    core, _ = train_core("residual", bundle, cfg["model"], cfg["train"], seed=int(cfg.get("seed", 0)))
    emulator = make_physical(core, bundle)
    space = bundle.space
    c0 = space.fiducial_vector()

    # Sensitivities + autodiff vs finite-difference check.
    J = abundance_jacobian(emulator, c0)
    J_fd = finite_difference_jacobian(emulator, c0)
    max_rel = float(np.max(np.abs((np.asarray(J) - np.asarray(J_fd)) / (np.abs(np.asarray(J)) + 1e-12))))
    logS = log_sensitivity(emulator, c0)
    print(f"[sens] max relative autodiff-vs-FD Jacobian discrepancy: {max_rel:.2e}")

    # Fisher forecast (with Planck-like prior on omega_b as an example).
    obs = ObservationalData.default()
    priors = {"omega_b": 0.0001} if "omega_b" in space.names else None
    report = forecast_report(emulator, c0, space, obs, priors=priors)
    print("[fisher] marginalized 1-sigma:")
    for n, s in zip(report["params"], report["marginal_sigma"]):
        print(f"   {n:>12s}: {s:.4g}")
    tracker.log_summary({"autodiff_fd_max_rel": max_rel, **report})

    plots.plot_sensitivity(logS, space.labels, bundle.target_names, save=tracker.path("sensitivity.png"))
    F = fisher_matrix(emulator, c0, obs, space=space, priors=priors)
    cov = parameter_covariance(F)
    plots.plot_fisher_corner(cov, c0, space.labels, save=tracker.path("fisher_corner.png"))
    print(f"[done] Fisher artifacts written to {args.out}")


if __name__ == "__main__":
    main()
