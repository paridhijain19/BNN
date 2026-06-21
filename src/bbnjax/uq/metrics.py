r"""Point-accuracy and probabilistic evaluation metrics.

All functions accept arrays of shape ``(N, T)`` (N samples, T targets) and, by
default, return a per-target vector; pass ``reduce=True`` for the scalar mean.
Probabilistic metrics take the predictive ``variance`` (not std).
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from jax.scipy.special import ndtri

_LOG2PI = jnp.log(2.0 * jnp.pi)


def _maybe_mean(x: Array, reduce: bool) -> Array:
    return jnp.mean(x) if reduce else x


def rmse(pred: Array, target: Array, *, reduce: bool = False) -> Array:
    return _maybe_mean(jnp.sqrt(jnp.mean((pred - target) ** 2, axis=0)), reduce)


def mae(pred: Array, target: Array, *, reduce: bool = False) -> Array:
    return _maybe_mean(jnp.mean(jnp.abs(pred - target), axis=0), reduce)


def max_abs_error(pred: Array, target: Array, *, reduce: bool = False) -> Array:
    return _maybe_mean(jnp.max(jnp.abs(pred - target), axis=0), reduce)


def r2_score(pred: Array, target: Array, *, reduce: bool = False) -> Array:
    ss_res = jnp.sum((target - pred) ** 2, axis=0)
    ss_tot = jnp.sum((target - jnp.mean(target, axis=0)) ** 2, axis=0) + 1e-30
    return _maybe_mean(1.0 - ss_res / ss_tot, reduce)


def gaussian_nll(mean: Array, var: Array, target: Array, *, reduce: bool = False) -> Array:
    var = jnp.clip(var, 1e-30, None)
    nll = 0.5 * (_LOG2PI + jnp.log(var) + (target - mean) ** 2 / var)
    return _maybe_mean(jnp.mean(nll, axis=0), reduce)


def standardized_residuals(mean: Array, var: Array, target: Array) -> Array:
    """z-scores ``(y - mu)/sigma``; should be ~N(0,1) if well calibrated."""
    return (target - mean) / jnp.sqrt(jnp.clip(var, 1e-30, None))


def picp(mean: Array, var: Array, target: Array, level: float = 0.68, *, reduce: bool = False) -> Array:
    """Prediction-Interval Coverage Probability for a central ``level`` interval."""
    z = jnp.abs(standardized_residuals(mean, var, target))
    half = ndtri(0.5 * (1.0 + level))
    covered = (z <= half).astype(jnp.float64)
    return _maybe_mean(jnp.mean(covered, axis=0), reduce)


def reliability_curve(
    mean: Array, var: Array, target: Array, levels: Array | None = None
) -> tuple[Array, Array]:
    """Return ``(expected, observed)`` central-coverage curves (averaged over targets).

    A perfectly calibrated emulator lies on the diagonal ``observed == expected``.
    """
    if levels is None:
        levels = jnp.linspace(0.05, 0.95, 19)
    observed = jnp.stack([picp(mean, var, target, float(p), reduce=True) for p in levels])
    return levels, observed


def calibration_error(mean: Array, var: Array, target: Array, levels: Array | None = None) -> float:
    """Expected calibration error: mean ``|observed - expected|`` over levels."""
    expected, observed = reliability_curve(mean, var, target, levels)
    return float(jnp.mean(jnp.abs(observed - expected)))


def sharpness(var: Array, *, reduce: bool = False) -> Array:
    """Average predictive std (lower = sharper); only meaningful once calibrated."""
    return _maybe_mean(jnp.mean(jnp.sqrt(jnp.clip(var, 0.0, None)), axis=0), reduce)


def summary(
    mean: Array, target: Array, var: Array | None = None, target_names: list[str] | None = None
) -> dict:
    """Convenience bundle of the main metrics as a nested dict."""
    out: dict = {
        "rmse": [float(v) for v in jnp.atleast_1d(rmse(mean, target))],
        "mae": [float(v) for v in jnp.atleast_1d(mae(mean, target))],
        "max_abs_error": [float(v) for v in jnp.atleast_1d(max_abs_error(mean, target))],
        "r2": [float(v) for v in jnp.atleast_1d(r2_score(mean, target))],
    }
    if var is not None:
        out["nll"] = [float(v) for v in jnp.atleast_1d(gaussian_nll(mean, var, target))]
        out["picp_68"] = [float(v) for v in jnp.atleast_1d(picp(mean, var, target, 0.68))]
        out["picp_95"] = [float(v) for v in jnp.atleast_1d(picp(mean, var, target, 0.95))]
        out["calibration_error"] = calibration_error(mean, var, target)
    if target_names is not None:
        out["targets"] = list(target_names)
    return out
