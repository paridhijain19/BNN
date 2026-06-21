"""Publication-quality figures (matplotlib).

Every function accepts a ``save`` path and returns the Matplotlib ``Figure`` so
plots can be composed or embedded. A consistent, paper-ready style is applied via
:func:`set_style`.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from bbnjax.parameters import TARGET_LABELS


def set_style() -> None:
    """Apply a clean, paper-ready Matplotlib style."""
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.axisbelow": True,
            "legend.frameon": False,
            "lines.linewidth": 1.8,
        }
    )


def _save(fig, save: str | Path | None):
    if save is not None:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save)
    return fig


def _labels(names):
    return [TARGET_LABELS.get(n, n) for n in names]


def plot_training_curves(train_loss, val_loss=None, *, save=None):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(np.asarray(train_loss), label="train")
    if val_loss is not None and len(val_loss):
        ax.plot(np.asarray(val_loss), label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_yscale("log")
    ax.legend()
    ax.set_title("Training history")
    return _save(fig, save)


def plot_pred_vs_true(mean, target, target_names, *, save=None):
    mean = np.asarray(mean)
    target = np.asarray(target)
    n = mean.shape[1]
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.2))
    axes = np.atleast_1d(axes)
    for j, ax in enumerate(axes):
        lo = min(target[:, j].min(), mean[:, j].min())
        hi = max(target[:, j].max(), mean[:, j].max())
        ax.scatter(target[:, j], mean[:, j], s=8, alpha=0.5)
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_xlabel("truth")
        ax.set_ylabel("emulator")
        ax.set_title(_labels(target_names)[j])
    fig.suptitle("Predicted vs. true (standardized)")
    fig.tight_layout()
    return _save(fig, save)


def plot_standardized_residuals(z, target_names, *, save=None):
    z = np.asarray(z)
    n = z.shape[1]
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.0))
    axes = np.atleast_1d(axes)
    grid = np.linspace(-4, 4, 200)
    gauss = np.exp(-0.5 * grid**2) / np.sqrt(2 * np.pi)
    for j, ax in enumerate(axes):
        ax.hist(z[:, j], bins=40, density=True, alpha=0.6)
        ax.plot(grid, gauss, "k-", lw=1.5)
        ax.set_title(_labels(target_names)[j])
        ax.set_xlabel(r"$(y-\mu)/\sigma$")
    fig.suptitle("Standardized residuals (calibration check)")
    fig.tight_layout()
    return _save(fig, save)


def plot_reliability(expected, observed, *, save=None, label=None):
    expected = np.asarray(expected)
    observed = np.asarray(observed)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="ideal")
    ax.plot(expected, observed, "o-", label=label or "emulator")
    ax.set_xlabel("expected coverage")
    ax.set_ylabel("observed coverage")
    ax.set_title("Reliability diagram")
    ax.legend()
    return _save(fig, save)


def plot_active_learning(histories: dict, *, metric="test_rmse_mean", save=None):
    """Learning curves comparing acquisition strategies."""
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    for name, hist in histories.items():
        h = hist.as_dict() if hasattr(hist, "as_dict") else hist
        ax.plot(h["n_train"], h[metric], "o-", label=name)
    ax.set_xlabel("number of simulations (training set size)")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_yscale("log")
    ax.set_title("Active learning: simulation efficiency")
    ax.legend()
    return _save(fig, save)


def plot_sensitivity(log_sens, param_labels, target_names, *, save=None):
    """Grouped bar chart of dimensionless log-sensitivities ``d ln Y / d ln c``."""
    log_sens = np.asarray(log_sens)  # (T, D)
    T, D = log_sens.shape
    x = np.arange(D)
    width = 0.8 / T
    fig, ax = plt.subplots(figsize=(1.6 * D + 2, 4))
    for a in range(T):
        ax.bar(x + a * width, log_sens[a], width, label=_labels(target_names)[a])
    ax.set_xticks(x + width * (T - 1) / 2)
    ax.set_xticklabels(param_labels, rotation=20)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel(r"$\partial \ln Y / \partial \ln c$")
    ax.set_title("Abundance sensitivities (autodiff)")
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    return _save(fig, save)


def _ellipse_xy(cov2, center, nsig=1.0, num=200):
    vals, vecs = np.linalg.eigh(cov2)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    t = np.linspace(0, 2 * np.pi, num)
    circle = np.stack([np.cos(t), np.sin(t)])
    ell = vecs @ (np.sqrt(np.maximum(vals, 0))[:, None] * circle) * nsig
    return ell[0] + center[0], ell[1] + center[1]


def plot_fisher_corner(cov, fiducial, param_labels, *, save=None):
    """Corner plot of Fisher 1- and 2-sigma confidence ellipses."""
    cov = np.asarray(cov)
    fiducial = np.asarray(fiducial)
    D = cov.shape[0]
    fig, axes = plt.subplots(D, D, figsize=(2.2 * D, 2.2 * D))
    axes = np.atleast_2d(axes)
    for i in range(D):
        for j in range(D):
            ax = axes[i, j]
            if j > i:
                ax.axis("off")
                continue
            if i == j:
                s = np.sqrt(cov[i, i])
                xs = np.linspace(fiducial[i] - 4 * s, fiducial[i] + 4 * s, 200)
                ax.plot(xs, np.exp(-0.5 * ((xs - fiducial[i]) / s) ** 2))
                ax.set_yticks([])
            else:
                sub = cov[np.ix_([j, i], [j, i])]
                center = (fiducial[j], fiducial[i])
                for nsig, alpha in ((2.0, 0.25), (1.0, 0.5)):
                    x, y = _ellipse_xy(sub, center, nsig=nsig)
                    ax.fill(x, y, alpha=alpha, color="C0")
                ax.plot(*center, "k+", ms=8)
            if i == D - 1:
                ax.set_xlabel(param_labels[j])
            if j == 0 and i > 0:
                ax.set_ylabel(param_labels[i])
    fig.suptitle("Fisher forecast")
    fig.tight_layout()
    return _save(fig, save)


def plot_benchmark(sizes, linx_times, emu_times, *, save=None):
    fig, ax = plt.subplots(figsize=(5, 3.6))
    ax.plot(sizes, linx_times, "o-", label="LINX")
    ax.plot(sizes, emu_times, "s-", label="emulator")
    ax.set_xlabel("number of evaluations")
    ax.set_ylabel("wall time [s]")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("LINX vs. emulator throughput")
    ax.legend()
    return _save(fig, save)
