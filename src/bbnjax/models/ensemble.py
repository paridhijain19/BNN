r"""Deep ensembles for scalable epistemic + aleatoric uncertainty.

Following Lakshminarayanan et al. (2017), an ensemble of :math:`M` independently
initialized (optionally heteroscedastic) networks approximates the predictive
distribution by a Gaussian mixture. We report the mixture mean and variance:

.. math::
    \mu(x)=\tfrac1M\sum_m \mu_m(x),\qquad
    \sigma^2(x)=\underbrace{\tfrac1M\sum_m \sigma_m^2(x)}_{\text{aleatoric}}
        +\underbrace{\tfrac1M\sum_m(\mu_m(x)-\mu(x))^2}_{\text{epistemic}}.

The epistemic term grows away from the training data, which is exactly the
signal active learning exploits. Cheaper and more parallelizable than a GP for
large designs; complementary to :mod:`nn_gp`.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from bbnjax.models.mlp import MLPEmulator


class DeepEnsemble(eqx.Module):
    """Container of independently trained MLP members with mixture statistics."""

    members: tuple

    @classmethod
    def init(
        cls,
        in_dim: int,
        out_dim: int,
        n_members: int,
        *,
        width: int = 128,
        depth: int = 4,
        activation: str = "gelu",
        heteroscedastic: bool = True,
        key: Array,
    ) -> "DeepEnsemble":
        keys = jax.random.split(key, n_members)
        members = tuple(
            MLPEmulator(
                in_dim,
                out_dim,
                width=width,
                depth=depth,
                activation=activation,
                heteroscedastic=heteroscedastic,
                key=k,
            )
            for k in keys
        )
        return cls(members=members)

    def __call__(self, x: Array) -> Array:
        mean, _ = self.predict(x)
        return mean

    def predict(self, x: Array) -> tuple[Array, Array]:
        """Return mixture ``(mean, variance)`` ``(T,)`` for a single input."""
        means, alea = [], []
        for m in self.members:
            mu, log_var = m.mean_and_logvar(x)
            means.append(mu)
            alea.append(jnp.where(jnp.isfinite(log_var), jnp.exp(log_var), 0.0))
        means = jnp.stack(means)
        alea = jnp.stack(alea)
        mean = jnp.mean(means, axis=0)
        aleatoric = jnp.mean(alea, axis=0)
        epistemic = jnp.var(means, axis=0)
        return mean, aleatoric + epistemic

    def predict_batch(self, X: Array) -> tuple[Array, Array]:
        return jax.vmap(self.predict)(jnp.atleast_2d(X))
