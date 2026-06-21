from __future__ import annotations

import numpy as np

from bbnjax.inference.fisher import (
    ObservationalData,
    conditional_errors,
    fisher_matrix,
    marginal_errors,
    parameter_covariance,
)


def test_fisher_symmetric_psd(space, sim):
    emulator = lambda c: sim.predict(c, space)
    F = np.asarray(fisher_matrix(emulator, space.fiducial_vector(), ObservationalData.default()))
    assert np.allclose(F, F.T, atol=1e-8)
    eig = np.linalg.eigvalsh(F)
    assert np.all(eig > -1e-8)


def test_marginal_ge_conditional(space, sim):
    emulator = lambda c: sim.predict(c, space)
    F = fisher_matrix(emulator, space.fiducial_vector(), ObservationalData.default())
    marg = np.asarray(marginal_errors(F))
    cond = np.asarray(conditional_errors(F))
    # Marginalizing over other parameters cannot reduce the error.
    assert np.all(marg >= cond - 1e-8)


def test_priors_shrink_errors(space, sim):
    emulator = lambda c: sim.predict(c, space)
    c0 = space.fiducial_vector()
    obs = ObservationalData.default()
    F0 = fisher_matrix(emulator, c0, obs, space=space)
    F1 = fisher_matrix(emulator, c0, obs, space=space, priors={"omega_b": 1e-4})
    i = space.index("omega_b")
    s0 = float(marginal_errors(F0)[i])
    s1 = float(marginal_errors(F1)[i])
    assert s1 < s0
    assert parameter_covariance(F1).shape == (space.dim, space.dim)
