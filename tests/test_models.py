from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from bbnjax.models import DeepEnsemble, MLPEmulator, build_model, make_physical
from bbnjax.models.residual import LinearBaseline


def test_mlp_forward_shape(bundle):
    key = jax.random.PRNGKey(0)
    m = MLPEmulator(bundle.input_dim, bundle.output_dim, width=16, depth=2, key=key)
    X, _ = bundle.xy("train")
    assert m(X[0]).shape == (bundle.output_dim,)
    assert jax.vmap(m)(X).shape == (X.shape[0], bundle.output_dim)


def test_linear_baseline_reasonable(bundle):
    X, T = bundle.xy("train")
    base = LinearBaseline.fit(X, T)
    pred = jax.vmap(base)(X)
    # A linear fit should explain most variance of the smooth mock.
    ss_res = float(jnp.sum((T - pred) ** 2))
    ss_tot = float(jnp.sum((T - jnp.mean(T, axis=0)) ** 2))
    assert 1.0 - ss_res / ss_tot > 0.9


def test_residual_builds(bundle):
    X, T = bundle.xy("train")
    key = jax.random.PRNGKey(0)
    m = build_model("residual", bundle.input_dim, bundle.output_dim, {"width": 16, "depth": 2}, key=key, X_train=X, T_train=T)
    assert m(X[0]).shape == (bundle.output_dim,)


def test_residual_passes_input_gradients(bundle):
    # Regression guard: the frozen baseline must still contribute to input
    # gradients (no stop_gradient), else autodiff sensitivities are wrong.
    X, T = bundle.xy("train")
    key = jax.random.PRNGKey(0)
    m = build_model(
        "residual", bundle.input_dim, bundle.output_dim, {"width": 16, "depth": 2}, key=key, X_train=X, T_train=T
    )
    x0 = X[0]
    J_full = jax.jacfwd(m)(x0)
    J_mlp = jax.jacfwd(m.mlp)(x0) * m.residual_scale
    assert np.allclose(np.asarray(J_full - J_mlp), np.asarray(m.baseline.W), atol=1e-6)


def test_ensemble_variance_nonneg(bundle):
    key = jax.random.PRNGKey(0)
    ens = DeepEnsemble.init(bundle.input_dim, bundle.output_dim, 3, width=16, depth=2, key=key)
    X, _ = bundle.xy("train")
    mean, var = ens.predict_batch(X[:10])
    assert mean.shape == (10, bundle.output_dim)
    assert np.all(np.asarray(var) >= 0.0)


def test_physical_wrapper_roundtrip(bundle, sim, space):
    key = jax.random.PRNGKey(0)
    m = MLPEmulator(bundle.input_dim, bundle.output_dim, width=16, depth=2, key=key)
    pe = make_physical(m, bundle)
    c = space.fiducial_vector()
    y = pe(c)
    assert y.shape == (bundle.output_dim,)
    # Log targets must be positive after inverse transform.
    assert np.all(np.asarray(y) > 0)
