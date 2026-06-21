r"""Residual-learning emulator: a cheap physics/linear baseline + NN correction.

.. math::
    \hat T(x) = \underbrace{B(x)}_{\text{frozen baseline}} + s\cdot \mathrm{NN}(x).

Learning the *residual* of a strong baseline typically converges faster, needs
fewer simulations, and yields smaller errors than learning the full map from
scratch — the baseline absorbs the dominant (near-linear) dependence of BBN
abundances on parameters, leaving the NN to model curvature/cross-terms.

The default baseline is a closed-form **ridge linear fit** in standardized space
(an excellent approximation given the smooth power-law structure of BBN). The
baseline *parameters* are frozen during training by the optimizer's parameter
filter (see :func:`bbnjax.training.trainer._trainable_filter`), **not** by
``stop_gradient`` -- this is deliberate, so that *input* gradients
:math:`\partial \hat T/\partial x` still flow through the baseline. Stopping the
gradient here would zero out the dominant (near-linear) sensitivity and break the
autodiff Jacobians / Fisher forecasts.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jax import Array

from bbnjax.models.mlp import MLPEmulator


class LinearBaseline(eqx.Module):
    """Affine baseline ``B(x) = W x + b`` (fit closed-form, then frozen)."""

    W: Array
    b: Array

    @classmethod
    def fit(cls, X: Array, T: Array, ridge: float = 1e-6) -> "LinearBaseline":
        X = jnp.asarray(X)
        T = jnp.asarray(T)
        n, d = X.shape
        Xa = jnp.concatenate([X, jnp.ones((n, 1))], axis=1)
        A = Xa.T @ Xa + ridge * jnp.eye(d + 1)
        coef = jnp.linalg.solve(A, Xa.T @ T)  # (d+1, T)
        return cls(W=coef[:d].T, b=coef[d])

    def __call__(self, x: Array) -> Array:
        return self.W @ x + self.b


class ResidualEmulator(eqx.Module):
    """Frozen baseline + trainable NN residual correction."""

    baseline: LinearBaseline
    mlp: MLPEmulator
    residual_scale: float = eqx.field(static=True)

    def __init__(
        self,
        baseline: LinearBaseline,
        mlp: MLPEmulator,
        residual_scale: float = 1.0,
    ) -> None:
        self.baseline = baseline
        self.mlp = mlp
        self.residual_scale = residual_scale

    def __call__(self, x: Array) -> Array:
        # Baseline params are frozen via the trainer's parameter filter, so we do
        # NOT stop_gradient here: input gradients must pass through the baseline.
        return self.baseline(x) + self.residual_scale * self.mlp(x)


def build_residual(in_dim: int, out_dim: int, cfg, X_train: Array, T_train: Array, key: Array):
    """Fit the linear baseline on train data and wrap an MLP residual head."""
    baseline = LinearBaseline.fit(X_train, T_train)
    mlp = MLPEmulator(
        in_dim,
        out_dim,
        width=int(cfg.get("width", 128)),
        depth=int(cfg.get("depth", 4)),
        activation=cfg.get("activation", "gelu"),
        heteroscedastic=bool(cfg.get("heteroscedastic", False)),
        key=key,
    )
    return ResidualEmulator(baseline, mlp, residual_scale=float(cfg.get("residual_scale", 1.0)))
