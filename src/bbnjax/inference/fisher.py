r"""Fisher-matrix forecasting for BBN parameters using the differentiable emulator.

For a Gaussian likelihood with parameter-independent data covariance
:math:`\Sigma` (measurement errors, optionally plus emulator variance), the
Fisher information matrix is

.. math:: F_{ij} = J^\top \Sigma^{-1} J\Big|_{ij}
   = \sum_{ab}\frac{\partial Y_a}{\partial c_i}\,(\Sigma^{-1})_{ab}\,
     \frac{\partial Y_b}{\partial c_j},

with :math:`J=\partial Y/\partial c` from :mod:`sensitivity`. Parameter
covariance is :math:`F^{-1}`; marginalized 1-sigma errors are
:math:`\sqrt{(F^{-1})_{ii}}`. Gaussian priors add :math:`1/\sigma_{\pi,i}^2` to
the diagonal.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jax import Array

from bbnjax.inference.sensitivity import abundance_jacobian
from bbnjax.parameters import TARGETS, ParameterSpace


@dataclass
class ObservationalData:
    """Measured primordial abundances (linear units) and 1-sigma errors.

    Defaults collate widely-used measurements; ``He3H`` is poorly constrained and
    ``Li7H`` reflects the (discrepant) observed value, not the SBBN prediction.
    Override for your analysis.
    """

    names: tuple[str, ...]
    mean: np.ndarray
    sigma: np.ndarray

    @classmethod
    def default(cls) -> "ObservationalData":
        return cls(
            names=TARGETS,
            mean=np.array([0.2450, 2.527e-5, 1.1e-5, 1.6e-10]),
            sigma=np.array([0.0030, 0.030e-5, 0.2e-5, 0.3e-10]),
        )

    def covariance(self) -> Array:
        return jnp.diag(jnp.asarray(self.sigma) ** 2)


def fisher_matrix(
    emulator,
    c: Array,
    obs: ObservationalData | None = None,
    *,
    emulator_var: Array | None = None,
    space: ParameterSpace | None = None,
    priors: dict[str, float] | None = None,
) -> Array:
    """Fisher matrix ``(D, D)`` at fiducial ``c``.

    Parameters
    ----------
    emulator:
        Callable physical params ``(D,)`` -> abundances ``(T,)`` (differentiable).
    emulator_var:
        Optional per-target emulator predictive variance (linear units) added to
        the measurement covariance diagonal — propagating surrogate error into the
        forecast.
    priors:
        Optional ``name -> sigma`` Gaussian priors (requires ``space`` for index).
    """
    obs = obs or ObservationalData.default()
    c = jnp.asarray(c)
    J = abundance_jacobian(emulator, c)  # (T, D)

    cov = obs.covariance()
    if emulator_var is not None:
        cov = cov + jnp.diag(jnp.asarray(emulator_var))
    cov_inv = jnp.linalg.inv(cov)

    F = J.T @ cov_inv @ J

    if priors:
        if space is None:
            raise ValueError("Priors require a ParameterSpace to resolve indices.")
        add = jnp.zeros(F.shape[0])
        for name, sigma in priors.items():
            add = add.at[space.index(name)].add(1.0 / sigma**2)
        F = F + jnp.diag(add)
    return F


def parameter_covariance(F: Array) -> Array:
    """Invert the Fisher matrix (with a tiny jitter for conditioning)."""
    d = F.shape[0]
    return jnp.linalg.inv(F + 1e-12 * jnp.eye(d))


def marginal_errors(F: Array) -> Array:
    """Marginalized 1-sigma errors ``sqrt(diag(F^-1))`` ``(D,)``."""
    return jnp.sqrt(jnp.diag(parameter_covariance(F)))


def conditional_errors(F: Array) -> Array:
    """Conditional (all-others-fixed) errors ``1/sqrt(diag(F))``."""
    return 1.0 / jnp.sqrt(jnp.diag(F))


def correlation_matrix(cov: Array) -> Array:
    d = jnp.sqrt(jnp.diag(cov))
    return cov / jnp.outer(d, d)


def figure_of_merit(cov: Array, i: int, j: int) -> float:
    """DETF-style FoM for a parameter pair: ``1/sqrt(det Cov_2x2)``."""
    sub = jnp.array([[cov[i, i], cov[i, j]], [cov[j, i], cov[j, j]]])
    return float(1.0 / jnp.sqrt(jnp.linalg.det(sub)))


def forecast_report(
    emulator,
    c: Array,
    space: ParameterSpace,
    obs: ObservationalData | None = None,
    *,
    emulator_var: Array | None = None,
    priors: dict[str, float] | None = None,
) -> dict:
    """Human-readable forecast: marginalized/conditional errors + correlations."""
    F = fisher_matrix(emulator, c, obs, emulator_var=emulator_var, space=space, priors=priors)
    cov = parameter_covariance(F)
    return {
        "params": list(space.names),
        "fiducial": [float(v) for v in jnp.asarray(c)],
        "marginal_sigma": [float(v) for v in marginal_errors(F)],
        "conditional_sigma": [float(v) for v in conditional_errors(F)],
        "correlation": np.asarray(correlation_matrix(cov)).tolist(),
        "fisher": np.asarray(F).tolist(),
    }
