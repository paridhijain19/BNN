"""Stage 08: end-to-end demo that regenerates all publication figures.

Runs a small but complete pipeline on the mock simulator and writes figures to
``docs/figures/``. Intended as a fast, self-contained reproduction of the paper's
qualitative results (use larger configs + LINX for the real numbers).
"""

from __future__ import annotations

import argparse

import jax

from _common import ROOT, load_config, load_or_generate_bundle, train_core
from bbnjax.active_learning import compare_acquisitions
from bbnjax.inference.fisher import ObservationalData, fisher_matrix, parameter_covariance
from bbnjax.inference.sensitivity import log_sensitivity
from bbnjax.models import NNGPEmulator, make_physical
from bbnjax.uq import metrics
from bbnjax.uq.calibration import VarianceCalibrator
from bbnjax.viz import plots, set_style


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train_mlp.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_style()

    figdir = ROOT / "docs" / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    bundle = load_or_generate_bundle(cfg)

    # This is a fast demo (figures), not a science run: cap training length.
    demo_train = dict(cfg["train"])
    demo_train["epochs"] = min(int(cfg["train"].get("epochs", 250)), 250)
    cfg["train"] = demo_train

    # --- residual emulator: accuracy ---
    core, result = train_core("residual", bundle, cfg["model"], cfg["train"], seed=0)
    X_test, T_test = bundle.xy("test")
    pred = jax.vmap(core)(X_test)
    plots.plot_training_curves(result.train_loss, result.val_loss, save=figdir / "fig_training.png")
    plots.plot_pred_vs_true(pred, T_test, bundle.target_names, save=figdir / "fig_pred_vs_true.png")

    # --- NN+GP: uncertainty + calibration ---
    nn, _ = train_core("mlp", bundle, cfg["model"], cfg["train"], seed=1)
    X, T = bundle.xy("train")
    nngp = NNGPEmulator.fit(nn, X, T, gp_steps=150)
    mean, var = nngp.predict_batch(X_test)
    Xval, Tval = bundle.xy("val")
    mval, vval = nngp.predict_batch(Xval)
    calib = VarianceCalibrator.fit(mval, vval, Tval)
    e0, o0 = metrics.reliability_curve(mean, var, T_test)
    e1, o1 = metrics.reliability_curve(mean, calib.apply(var), T_test)
    fig = plots.plot_reliability(e0, o0, label="raw")
    fig.axes[0].plot(e1, o1, "s-", color="C1", label="calibrated")
    fig.axes[0].legend()
    fig.savefig(figdir / "fig_reliability.png")

    # --- sensitivities + Fisher ---
    emulator = make_physical(core, bundle)
    c0 = bundle.space.fiducial_vector()
    logS = log_sensitivity(emulator, c0)
    plots.plot_sensitivity(logS, bundle.space.labels, bundle.target_names, save=figdir / "fig_sensitivity.png")
    F = fisher_matrix(emulator, c0, ObservationalData.default(), space=bundle.space)
    plots.plot_fisher_corner(parameter_covariance(F), c0, bundle.space.labels, save=figdir / "fig_fisher.png")

    # --- active learning (small) ---
    al_cfg = load_config("configs/active_learning.yaml")
    al = dict(al_cfg["active_learning"])
    al.update({"init_size": 128, "rounds": 4, "batch_per_round": 48, "pool_size": 1024, "retrain_epochs": 120})
    al_cfg["active_learning"] = al
    histories = compare_acquisitions(al_cfg, acquisitions=("max_variance", "random"))
    plots.plot_active_learning(histories, metric="test_rmse_mean", save=figdir / "fig_active_learning.png")

    print(f"[done] figures written to {figdir}")


if __name__ == "__main__":
    main()
