from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from bbnjax.sampling import latin_hypercube, make_design, sobol_design, uniform_design


def test_shapes_and_range():
    for design in (
        sobol_design(64, 4, seed=0),
        latin_hypercube(jax.random.PRNGKey(0), 64, 4),
        uniform_design(jax.random.PRNGKey(0), 64, 4),
    ):
        assert design.shape == (64, 4)
        assert float(jnp.min(design)) >= 0.0
        assert float(jnp.max(design)) < 1.0 + 1e-9


def test_sobol_deterministic():
    a = sobol_design(32, 3, seed=7)
    b = sobol_design(32, 3, seed=7)
    assert np.allclose(np.asarray(a), np.asarray(b))


def test_lhs_strata_coverage():
    # Each axis should have exactly one sample per 1/n stratum.
    n, d = 50, 4
    pts = np.asarray(latin_hypercube(jax.random.PRNGKey(3), n, d))
    for j in range(d):
        strata = np.floor(pts[:, j] * n).astype(int)
        assert len(np.unique(strata)) == n


def test_make_design_dispatch():
    assert make_design("sobol", 8, 2, seed=0).shape == (8, 2)
    assert make_design("lhs", 8, 2, seed=0).shape == (8, 2)
    assert make_design("uniform", 8, 2, seed=0).shape == (8, 2)
