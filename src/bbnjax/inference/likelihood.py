r"""Differentiable BBN likelihood and a Cobaya-compatible wrapper.

The Gaussian log-likelihood for selected observables :math:`a\in\mathcal O` is

.. math::
    \ln\mathcal L(c) = -\tfrac12\sum_{a\in\mathcal O}
      \left[\frac{(Y_a(c)-\hat Y_a)^2}{\sigma_a^2+\sigma_{\mathrm{emu},a}^2}
      + \ln\!\big(2\pi(\sigma_a^2+\sigma_{\mathrm{emu},a}^2)\big)\right],

where :math:`Y_a(c)` is the emulator prediction and the emulator variance
:math:`\sigma_{\mathrm{emu}}^2` can optionally be folded into the error budget.
By default only the well-measured :math:`Y_p` and D/H are used (standard
"BBN-only" analyses), avoiding the :math:`^3`He and :math:`^7`Li systematics.

Because everything is JAX, :func:`grad_log_prob` gives exact score functions for
HMC/NUTS (e.g. via blackjax) and the Cobaya wrapper exposes ``logp`` for
gradient-free or gradient-based Cobaya samplers.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from bbnjax.inference.fisher import ObservationalData
from bbnjax.parameters import TARGETS, ParameterSpace


class BBNLikelihood:
    """Gaussian likelihood over selected primordial abundances."""

    def __init__(
        self,
        emulator,
        space: ParameterSpace,
        obs: ObservationalData | None = None,
        *,
        use: tuple[str, ...] = ("Yp", "DH"),
        include_emulator_var: bool = True,
    ) -> None:
        obs = obs or ObservationalData.default()
        self.emulator = emulator
        self.space = space
        self.use = tuple(use)
        self.include_emulator_var = include_emulator_var
        self._idx = jnp.asarray([TARGETS.index(t) for t in use])
        self._obs_mean = jnp.asarray(obs.mean)[self._idx]
        self._obs_var = jnp.asarray(obs.sigma)[self._idx] ** 2

    def _predict(self, c: Array) -> tuple[Array, Array]:
        if self.include_emulator_var and hasattr(self.emulator, "predict"):
            mean, std = self.emulator.predict(c)
            return mean[self._idx], (std[self._idx]) ** 2
        mean = self.emulator(c)
        return mean[self._idx], jnp.zeros_like(self._obs_mean)

    def log_prob(self, c: Array) -> Array:
        """Scalar log-likelihood at physical parameters ``c`` (differentiable)."""
        c = jnp.asarray(c)
        mean, emu_var = self._predict(c)
        var = self._obs_var + emu_var
        resid = mean - self._obs_mean
        return -0.5 * jnp.sum(resid**2 / var + jnp.log(2.0 * jnp.pi * var))

    def chi2(self, c: Array) -> Array:
        c = jnp.asarray(c)
        mean, emu_var = self._predict(c)
        var = self._obs_var + emu_var
        return jnp.sum((mean - self._obs_mean) ** 2 / var)

    def grad_log_prob(self, c: Array) -> Array:
        """Exact gradient of the log-likelihood (for HMC/NUTS)."""
        return jax.grad(self.log_prob)(jnp.asarray(c))


def make_cobaya_likelihood(
    emulator,
    space: ParameterSpace,
    obs: ObservationalData | None = None,
    *,
    use: tuple[str, ...] = ("Yp", "DH"),
    include_emulator_var: bool = True,
):
    """Build a Cobaya *external-function* likelihood and its info block.

    Returns ``(logp, info)`` where ``logp(**params)`` accepts the sampled
    parameters by name and ``info`` is a ready-to-use Cobaya ``likelihood`` dict::

        from cobaya.run import run
        logp, info = make_cobaya_likelihood(emulator, space)
        run({"likelihood": {"bbn": info}, "params": {...}, "sampler": {...}})
    """
    like = BBNLikelihood(emulator, space, obs, use=use, include_emulator_var=include_emulator_var)
    names = list(space.names)

    def logp(**params) -> float:
        c = jnp.asarray([params[n] for n in names])
        return float(like.log_prob(c))

    info = {"external": logp, "input_params": names}
    return logp, info


def make_cobaya_class():
    """Return a Cobaya ``Likelihood`` subclass (only if cobaya is installed).

    Usage: subclass-style integration where the emulator is loaded in
    ``initialize``. Kept as a factory so importing this module never requires
    cobaya.
    """
    try:
        from cobaya.likelihood import Likelihood
    except Exception as exc:  # pragma: no cover - optional dep
        raise ImportError("cobaya is not installed; use make_cobaya_likelihood instead.") from exc

    class BBNEmulatorLikelihood(Likelihood):
        emulator_path: str = ""
        use: tuple = ("Yp", "DH")

        def initialize(self):
            from bbnjax.tracking.experiment import load_emulator

            self._space, emulator = load_emulator(self.emulator_path)
            self._like = BBNLikelihood(emulator, self._space, use=tuple(self.use))

        def get_requirements(self):
            return {}

        def logp(self, **params):
            c = np.array([params[n] for n in self._space.names])
            return float(self._like.log_prob(c))

    return BBNEmulatorLikelihood
