from __future__ import annotations

import jax
import jax.numpy as jnp

from bbnjax.models import MLPEmulator
from bbnjax.training import train_model
from bbnjax.training.losses import mse_loss


def test_training_reduces_loss(bundle):
    X, T = bundle.xy("train")
    Xv, Tv = bundle.xy("val")
    key = jax.random.PRNGKey(0)
    model = MLPEmulator(bundle.input_dim, bundle.output_dim, width=32, depth=2, key=key)
    init_loss = float(mse_loss(model, X, T))
    res = train_model(
        model, X, T, X_val=Xv, T_val=Tv, epochs=40, batch_size=64, lr=3e-3, key=key
    )
    final_loss = float(mse_loss(res.model, X, T))
    assert final_loss < 0.5 * init_loss
    assert res.best_val < float("inf")


def test_baseline_frozen_in_residual(bundle):
    from bbnjax.models import build_residual

    X, T = bundle.xy("train")
    key = jax.random.PRNGKey(0)
    model = build_residual(bundle.input_dim, bundle.output_dim, {"width": 16, "depth": 2}, X, T, key)
    W0 = jnp.asarray(model.baseline.W)
    res = train_model(model, X, T, epochs=20, batch_size=64, lr=3e-3, key=key)
    assert jnp.allclose(res.model.baseline.W, W0)  # baseline must not change
