"""Space-filling experimental designs on the unit cube :math:`[0,1]^D`.

Provides Sobol' (low-discrepancy quasi-random) and Latin-Hypercube designs, the
two work-horses for building emulator training sets. Sobol' sequences give
excellent uniformity and are the recommended default; LHS is a robust, fully
JAX-native alternative with good marginal coverage.

All functions return ``jax.numpy`` arrays of shape ``(n, d)`` with entries in
``[0, 1)``. Map to physical parameters with :meth:`ParameterSpace.from_unit`.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array


def latin_hypercube(key: Array, n: int, d: int, *, centered: bool = False) -> Array:
    """Latin-Hypercube sample, fully JAX-native.

    Each of the ``d`` axes is partitioned into ``n`` equal strata; exactly one
    sample falls in each stratum per axis, with an independent random permutation
    per axis to decorrelate them.

    Parameters
    ----------
    key:
        ``jax.random`` key.
    n, d:
        Number of points and dimensions.
    centered:
        If ``True`` place points at stratum centers (deterministic given perms);
        otherwise jitter uniformly within each stratum.
    """
    perm_key, jit_key = jax.random.split(key)
    # One independent permutation of {0..n-1} per dimension.
    perm_keys = jax.random.split(perm_key, d)
    perms = jnp.stack([jax.random.permutation(k, n) for k in perm_keys], axis=1)  # (n, d)
    if centered:
        offsets = jnp.full((n, d), 0.5)
    else:
        offsets = jax.random.uniform(jit_key, shape=(n, d))
    return (perms + offsets) / n


def sobol_design(n: int, d: int, *, seed: int = 0, scramble: bool = True) -> Array:
    """Sobol' low-discrepancy sequence via ``scipy.stats.qmc`` (returned as JAX array).

    Sobol' is deterministic given ``seed`` and ``scramble``. ``n`` need not be a
    power of two, but powers of two give the best balance properties.

    Notes
    -----
    Sobol' construction is inherently sequential and CPU-side; we generate it
    with SciPy's well-tested implementation and move the result onto the JAX
    device. This is the only non-JAX numerical step in the data pipeline.
    """
    try:
        from scipy.stats import qmc
    except Exception as exc:  # pragma: no cover - optional dep path
        raise ImportError(
            "Sobol' designs require scipy. Install scipy or use design='lhs'."
        ) from exc

    sampler = qmc.Sobol(d=d, scramble=scramble, seed=seed)
    with warnings.catch_warnings():
        # n need not be a power of two; SciPy warns about balance properties only.
        warnings.simplefilter("ignore", UserWarning)
        points = sampler.random(n)
    return jnp.asarray(np.asarray(points))


def uniform_design(key: Array, n: int, d: int) -> Array:
    """Plain i.i.d. uniform design (baseline; poor space-filling)."""
    return jax.random.uniform(key, shape=(n, d))


def make_design(
    kind: str,
    n: int,
    d: int,
    *,
    key: Array | None = None,
    seed: int = 0,
    scramble: bool = True,
) -> Array:
    """Dispatch to a named design. ``kind`` in {``sobol``, ``lhs``, ``uniform``}."""
    kind = kind.lower()
    if kind == "sobol":
        return sobol_design(n, d, seed=seed, scramble=scramble)
    if kind in ("lhs", "latin_hypercube", "latin"):
        if key is None:
            key = jax.random.PRNGKey(seed)
        return latin_hypercube(key, n, d)
    if kind == "uniform":
        if key is None:
            key = jax.random.PRNGKey(seed)
        return uniform_design(key, n, d)
    raise ValueError(f"Unknown design kind: {kind!r}")


def discrepancy(points: Array) -> float:
    """Centered L2 discrepancy (lower = more uniform); diagnostic for designs."""
    try:
        from scipy.stats import qmc
    except Exception as exc:  # pragma: no cover
        raise ImportError("discrepancy requires scipy") from exc
    return float(qmc.discrepancy(np.asarray(points), method="CD"))
