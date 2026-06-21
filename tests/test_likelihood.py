from __future__ import annotations

import numpy as np

from bbnjax.inference.fisher import ObservationalData
from bbnjax.inference.likelihood import BBNLikelihood, make_cobaya_likelihood


def test_log_prob_and_grad_finite(space, sim):
    emulator = lambda c: sim.predict(c, space)
    like = BBNLikelihood(emulator, space, use=("Yp", "DH"), include_emulator_var=False)
    c0 = space.fiducial_vector()
    lp = float(like.log_prob(c0))
    g = np.asarray(like.grad_log_prob(c0))
    assert np.isfinite(lp)
    assert g.shape == (space.dim,)
    assert np.all(np.isfinite(g))


def test_chi2_nonnegative(space, sim):
    emulator = lambda c: sim.predict(c, space)
    like = BBNLikelihood(emulator, space, include_emulator_var=False)
    assert float(like.chi2(space.fiducial_vector())) >= 0.0


def test_cobaya_factory(space, sim):
    emulator = lambda c: sim.predict(c, space)
    logp, info = make_cobaya_likelihood(emulator, space, ObservationalData.default())
    params = {n: float(v) for n, v in zip(space.names, np.asarray(space.fiducial_vector()))}
    val = logp(**params)
    assert np.isfinite(val)
    assert info["input_params"] == list(space.names)
