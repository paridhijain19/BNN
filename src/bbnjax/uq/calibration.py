r"""Predictive-uncertainty calibration.

Emulator variances are often mis-scaled (a GP fit on residuals, or an ensemble,
rarely produces perfectly calibrated error bars). We rescale predictive variances
by a per-target factor :math:`s_t^2`,

.. math:: \tilde\sigma_t^2(x) = s_t^2\,\sigma_t^2(x),

chosen to minimize the Gaussian NLL on a held-out calibration set. The minimizer
is available in closed form,

.. math:: s_t^2 = \frac1N\sum_i \frac{(y_{it}-\mu_{it})^2}{\sigma_{it}^2},

i.e. the mean squared *standardized residual* (well-calibrated => :math:`s_t=1`).
A :class:`VarianceCalibrator` is an Equinox module so it composes with the rest
of the differentiable pipeline.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jax import Array


class VarianceCalibrator(eqx.Module):
    """Per-target variance rescaling ``var -> scale**2 * var``."""

    scale: Array  # (T,) std-multipliers

    @classmethod
    def fit(cls, mean: Array, var: Array, target: Array) -> "VarianceCalibrator":
        var = jnp.clip(var, 1e-30, None)
        s2 = jnp.mean((target - mean) ** 2 / var, axis=0)
        return cls(scale=jnp.sqrt(jnp.clip(s2, 1e-12, None)))

    def apply(self, var: Array) -> Array:
        return var * self.scale**2

    def __call__(self, var: Array) -> Array:
        return self.apply(var)


def calibrate_on(predict_fn, X_cal: Array, T_cal: Array) -> VarianceCalibrator:
    """Fit a calibrator using ``predict_fn(X) -> (mean, var)`` on calibration data."""
    mean, var = predict_fn(X_cal)
    return VarianceCalibrator.fit(mean, var, T_cal)
