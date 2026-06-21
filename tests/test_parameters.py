from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from bbnjax.parameters import (
    ParameterSpace,
    eta10_to_omega_b,
    omega_b_to_eta10,
)


def test_unit_roundtrip(space):
    u = jnp.linspace(0.1, 0.9, space.dim)
    phys = space.from_unit(u)
    back = space.to_unit(phys)
    assert np.allclose(np.asarray(back), np.asarray(u), atol=1e-10)


def test_bounds_and_fiducial(space):
    lo, hi = space.lo(), space.hi()
    assert np.all(np.asarray(lo) < np.asarray(hi))
    fid = space.fiducial_vector()
    assert np.all(np.asarray(fid) >= np.asarray(lo))
    assert np.all(np.asarray(fid) <= np.asarray(hi))


def test_eta_conversion():
    omega_b = 0.02237
    eta10 = omega_b_to_eta10(omega_b)
    assert np.isclose(float(eta10_to_omega_b(eta10)), omega_b, rtol=1e-12)
    assert 5.5 < float(eta10) < 6.5  # standard BBN value ~6.1


def test_from_config():
    cfg = {
        "active": ["omega_b", "N_eff"],
        "bounds": {"omega_b": [0.02, 0.024], "N_eff": [2.5, 3.5]},
        "fiducial": {"omega_b": 0.022, "N_eff": 3.0},
    }
    sp = ParameterSpace.from_config(cfg)
    assert sp.names == ("omega_b", "N_eff")
    assert sp.dim == 2
    assert sp.index("N_eff") == 1
