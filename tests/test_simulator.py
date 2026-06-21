from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def test_fiducial_abundances(space, sim):
    Y = np.asarray(sim.predict(space.fiducial_vector(), space))
    assert Y.shape == (4,)
    Yp, DH, He3H, Li7H = Y
    assert 0.24 < Yp < 0.25
    assert 2.0e-5 < DH < 3.0e-5
    assert 0.8e-5 < He3H < 1.3e-5
    assert 3e-10 < Li7H < 7e-10


def test_batch_matches_single(space, sim):
    C = space.from_unit(jnp.array([[0.2, 0.4, 0.6, 0.8], [0.5, 0.5, 0.5, 0.5]]))
    batch = np.asarray(sim.predict_batch(C, space))
    singles = np.stack([np.asarray(sim.predict(C[i], space)) for i in range(C.shape[0])])
    assert np.allclose(batch, singles, rtol=1e-8)


def test_differentiable(space, sim):
    c = space.fiducial_vector()
    J = jax.jacfwd(lambda x: sim.predict(x, space))(c)
    assert J.shape == (4, space.dim)
    assert np.all(np.isfinite(np.asarray(J)))
    # D/H should decrease with omega_b (index 0): dDH/domega_b < 0.
    assert float(J[1, 0]) < 0.0


def test_monotonic_yp_in_neff(space, sim):
    # Yp increases with N_eff.
    lo = sim.predict(space.from_unit(jnp.array([0.5, 0.1, 0.5, 0.5])), space)[0]
    hi = sim.predict(space.from_unit(jnp.array([0.5, 0.9, 0.5, 0.5])), space)[0]
    assert float(hi) > float(lo)
