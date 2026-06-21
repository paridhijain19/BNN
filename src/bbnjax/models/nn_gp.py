r"""NN + Gaussian-Process residual emulator (the UQ work-horse).

A trained neural network supplies a fast, smooth mean :math:`m_\mathrm{NN}(x)`.
An exact GP is then fit to the NN's *residuals* on the training set,

.. math:: r_i = T_i - m_\mathrm{NN}(x_i),

so the final emulator is

.. math::
    \hat T(x) = m_\mathrm{NN}(x) + \mu_\mathrm{GP}(x), \qquad
    \mathrm{Var}[\hat T(x)] = \sigma^2_\mathrm{GP}(x).

This hybrid keeps the NN's flexibility and speed while inheriting the GP's
principled, input-dependent error bars — which both calibrate predictions and
drive active learning. Everything is differentiable in ``x``.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from bbnjax.models.gp import MultiOutputGP, fit_gp


class NNGPEmulator(eqx.Module):
    """Neural-network mean with a GP residual correction and predictive variance."""

    nn: eqx.Module
    gp: MultiOutputGP

    @classmethod
    def fit(
        cls,
        nn: eqx.Module,
        X_train: Array,
        T_train: Array,
        *,
        kernel: str = "matern52",
        noise: float = 1e-3,
        gp_steps: int = 300,
        gp_lr: float = 1e-2,
    ) -> "NNGPEmulator":
        nn_pred = jax.vmap(nn)(X_train)
        residual = jnp.asarray(T_train) - nn_pred
        gp = MultiOutputGP.init(X_train, residual, kernel=kernel, noise=noise)
        gp, _ = fit_gp(gp, steps=gp_steps, lr=gp_lr)
        return cls(nn=nn, gp=gp)

    def __call__(self, x: Array) -> Array:
        """Mean prediction ``(T,)`` for a single standardized input."""
        mean, _ = self.predict(x)
        return mean

    def predict(self, x: Array) -> tuple[Array, Array]:
        """Return ``(mean, variance)`` ``(T,)`` for a single input ``x``."""
        nn_mean = self.nn(x)
        gp_mean, gp_var = self.gp.predict(jnp.atleast_2d(x))
        return nn_mean + gp_mean[0], gp_var[0]

    def predict_batch(self, X: Array) -> tuple[Array, Array]:
        """Batched ``(mean, variance)`` for ``X`` of shape ``(n, D)``."""
        X = jnp.atleast_2d(X)
        nn_mean = jax.vmap(self.nn)(X)
        gp_mean, gp_var = self.gp.predict(X)
        return nn_mean + gp_mean, gp_var
