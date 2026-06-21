r"""Loss functions for emulator training (all in standardized target space).

* :func:`mse_loss` -- homoscedastic regression (plain MLP / residual).
* :func:`gaussian_nll_loss` -- heteroscedastic negative log-likelihood
  :math:`\tfrac12\big[\log\sigma^2 + (y-\mu)^2/\sigma^2\big]` for ensemble members
  that predict their own variance (Nix & Weigend 1994; Lakshminarayanan 2017).
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array


def mse_loss(model: eqx.Module, X: Array, T: Array) -> Array:
    """Mean squared error of ``model(x)`` vs targets ``T`` over a batch."""
    preds = jax.vmap(model)(X)
    return jnp.mean((preds - T) ** 2)


def gaussian_nll_loss(model: eqx.Module, X: Array, T: Array, beta: float = 0.0) -> Array:
    r"""Heteroscedastic Gaussian NLL using ``model.mean_and_logvar``.

    ``beta`` in [0,1] applies the 'beta-NLL' variance-weighting of Seitzer et al.
    (2022) which stabilizes early training (0 = standard NLL).
    """

    def per_example(x, t):
        mean, log_var = model.mean_and_logvar(x)
        inv_var = jnp.exp(-log_var)
        nll = 0.5 * (log_var + (t - mean) ** 2 * inv_var)
        if beta > 0.0:
            weight = jax.lax.stop_gradient(jnp.exp(log_var) ** beta)
            nll = nll * weight
        return jnp.sum(nll)

    return jnp.mean(jax.vmap(per_example)(X, T))
