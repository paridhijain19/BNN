r"""Exact abundance sensitivities via automatic differentiation.

Because the emulator is differentiable end to end, the response of every
abundance to every cosmological parameter is available *exactly* (to machine
precision) and cheaply with :func:`jax.jacfwd` / :func:`jax.jacrev` — no finite
differencing, no simulator re-runs.

* :func:`abundance_jacobian` -- :math:`J_{ai}=\partial Y_a/\partial c_i`.
* :func:`log_sensitivity` -- dimensionless log-derivatives
  :math:`\partial\ln Y_a/\partial\ln c_i`, the standard way to report and compare
  BBN parameter dependences.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array


def abundance_jacobian(emulator, c: Array) -> Array:
    """Jacobian ``J[a, i] = dY_a/dc_i`` at physical parameters ``c`` -> ``(T, D)``.

    ``emulator`` is any callable mapping physical params ``(D,)`` to abundances
    ``(T,)`` (e.g. :class:`PhysicalEmulator` or a simulator-derived function).
    Forward-mode is efficient here since ``D`` (params) is small.
    """
    return jax.jacfwd(lambda x: emulator(x))(jnp.asarray(c))


def log_sensitivity(emulator, c: Array) -> Array:
    """Dimensionless ``d ln Y_a / d ln c_i`` at ``c`` -> ``(T, D)``."""
    c = jnp.asarray(c)
    y = emulator(c)
    J = abundance_jacobian(emulator, c)
    return J * (c[None, :] / y[:, None])


def batched_jacobian(emulator, C: Array) -> Array:
    """Stack of Jacobians over a batch ``C`` of shape ``(N, D)`` -> ``(N, T, D)``."""
    return jax.vmap(lambda c: abundance_jacobian(emulator, c))(jnp.atleast_2d(C))


def finite_difference_jacobian(emulator, c: Array, eps: float = 1e-4) -> Array:
    """Central-difference Jacobian for validating the autodiff result."""
    c = jnp.asarray(c)
    d = c.shape[0]

    def col(i):
        step = jnp.zeros(d).at[i].set(eps * jnp.maximum(jnp.abs(c[i]), 1.0))
        return (emulator(c + step) - emulator(c - step)) / (2.0 * step[i])

    return jnp.stack([col(i) for i in range(d)], axis=1)
