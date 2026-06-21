from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from bbnjax.models.gp import GPModel, MultiOutputGP, fit_gp


def _toy(n=40, d=2, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n, d))
    y = np.sin(3 * X[:, 0]) + 0.5 * X[:, 1] ** 2
    return jnp.asarray(X), jnp.asarray(y)


def test_lml_improves_with_fit():
    X, y = _toy()
    gp = GPModel.init(X, y, noise=1e-3)
    before = float(gp.log_marginal_likelihood())
    gp, _ = fit_gp(gp, steps=80, lr=5e-2)
    after = float(gp.log_marginal_likelihood())
    assert after >= before - 1e-6


def test_predict_interpolates_training_points():
    X, y = _toy(n=30)
    gp = GPModel.init(X, y, noise=1e-6)
    gp, _ = fit_gp(gp, steps=120, lr=5e-2)
    mean, var = gp.predict(X)
    assert np.allclose(np.asarray(mean), np.asarray(y), atol=1e-2)
    assert np.all(np.asarray(var) > 0)


def test_multioutput_shapes():
    X, y = _toy()
    Y = jnp.stack([y, 2 * y], axis=-1)
    gp = MultiOutputGP.init(X, Y, noise=1e-3)
    mean, var = gp.predict(X[:5])
    assert mean.shape == (5, 2)
    assert var.shape == (5, 2)
