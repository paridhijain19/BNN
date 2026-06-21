r"""Acquisition functions for active learning.

Given an emulator's predictive uncertainty over a candidate *pool*, an
acquisition function scores how informative each candidate would be. We provide:

* :func:`score_max_variance` -- total predictive variance (uncertainty sampling).
* :func:`score_bald` -- BALD / mutual information (epistemic-only, Gaussian
  approximation): :math:`\mathcal I \approx \tfrac12\sum_t\log(\sigma^2_t/\sigma^2_{a,t})`.
* :func:`gp_alc_scores` -- Active Learning Cohn: expected *integrated* variance
  reduction over a reference set, computed in closed form for a GP.

Batch selection (:func:`select_batch`) supports a greedy diversity penalty so a
round does not collapse onto one region of parameter space.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from bbnjax.models.gp import GPModel, MultiOutputGP


def score_max_variance(var: Array, weights: Array | None = None) -> Array:
    """Sum of per-target predictive variances; ``var`` is ``(N, T)`` -> ``(N,)``."""
    if weights is not None:
        var = var * weights
    return jnp.sum(var, axis=-1)


def score_bald(total_var: Array, aleatoric_var: Array, eps: float = 1e-12) -> Array:
    """Gaussian-approx mutual information between prediction and parameters."""
    ratio = (total_var + eps) / (aleatoric_var + eps)
    return 0.5 * jnp.sum(jnp.log(jnp.clip(ratio, 1.0, None)), axis=-1)


def _gp_alc_single(gp: GPModel, pool: Array, ref: Array) -> Array:
    """Integrated variance reduction for one GP over a reference set.

    For a candidate ``x_c`` the predictive-variance reduction at reference point
    ``x_r`` is ``k(x_r, x_c)^2 / (k(x_c, x_c) + noise)`` (rank-1 GP update);
    summing over ``x_r`` gives the integrated reduction (ALC criterion).
    """
    ls = jnp.exp(gp.log_lengthscale)
    sv = jnp.exp(gp.log_signal)
    noise = jnp.exp(gp.log_noise)
    k_rc = gp._kfun(ref, pool, ls, sv)  # (R, N)
    k_cc = sv + noise  # stationary diagonal
    reduction = jnp.sum(k_rc**2, axis=0) / k_cc
    return reduction


def gp_alc_scores(gp, pool: Array, ref: Array) -> Array:
    """ALC scores summed over GP outputs (handles GPModel or MultiOutputGP)."""
    if isinstance(gp, MultiOutputGP):
        return jnp.sum(jnp.stack([_gp_alc_single(g, pool, ref) for g in gp.gps]), axis=0)
    return _gp_alc_single(gp, pool, ref)


def select_batch(
    scores: Array,
    k: int,
    *,
    pool_points: Array | None = None,
    diversity: float = 0.0,
    length_scale: float = 0.25,
) -> Array:
    """Select indices of the top-``k`` candidates.

    If ``diversity > 0`` and ``pool_points`` is given, use greedy selection that
    down-weights candidates close (in scaled Euclidean distance) to already
    chosen points, encouraging a space-filling batch.
    """
    scores = jnp.asarray(scores)
    if diversity <= 0.0 or pool_points is None:
        return jnp.argsort(scores)[::-1][:k]

    pool_points = jnp.asarray(pool_points)
    chosen: list[int] = []
    remaining = jnp.array(scores)
    for _ in range(min(k, scores.shape[0])):
        idx = int(jnp.argmax(remaining))
        chosen.append(idx)
        d2 = jnp.sum((pool_points - pool_points[idx]) ** 2, axis=-1) / (length_scale**2)
        penalty = diversity * jnp.exp(-0.5 * d2)
        remaining = remaining - penalty * jnp.max(scores)
        remaining = remaining.at[idx].set(-jnp.inf)
    return jnp.asarray(chosen)


def random_scores(key: Array, n: int) -> Array:
    """Uniform random scores (the active-learning *baseline*)."""
    return jax.random.uniform(key, shape=(n,))
