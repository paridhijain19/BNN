from __future__ import annotations

import numpy as np

from bbnjax.inference.sensitivity import (
    abundance_jacobian,
    finite_difference_jacobian,
    log_sensitivity,
)


def test_autodiff_matches_finite_difference(space, sim):
    emulator = lambda c: sim.predict(c, space)
    c = space.fiducial_vector()
    J = np.asarray(abundance_jacobian(emulator, c))
    J_fd = np.asarray(finite_difference_jacobian(emulator, c, eps=1e-5))
    rel = np.abs(J - J_fd) / (np.abs(J) + 1e-12)
    assert np.max(rel) < 1e-3


def test_log_sensitivity_signs(space, sim):
    emulator = lambda c: sim.predict(c, space)
    logS = np.asarray(log_sensitivity(emulator, space.fiducial_vector()))
    # d ln(D/H) / d ln(omega_b) should be strongly negative (~ -1.6).
    i_ob = space.index("omega_b")
    assert logS[1, i_ob] < -1.0
