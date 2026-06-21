from __future__ import annotations

import numpy as np

from bbnjax.uq import metrics
from bbnjax.uq.calibration import VarianceCalibrator


def _gaussian_preds(n=4000, t=2, scale=2.0, seed=0):
    rng = np.random.default_rng(seed)
    mean = rng.normal(size=(n, t))
    true_std = 0.3
    target = mean + rng.normal(scale=true_std, size=(n, t))
    var = np.full((n, t), (true_std / scale) ** 2)  # deliberately mis-scaled
    return mean, var, target, true_std


def test_calibration_fixes_scale():
    import jax.numpy as jnp

    mean, var, target, _ = _gaussian_preds()
    before = metrics.calibration_error(jnp.asarray(mean), jnp.asarray(var), jnp.asarray(target))
    calib = VarianceCalibrator.fit(jnp.asarray(mean), jnp.asarray(var), jnp.asarray(target))
    after = metrics.calibration_error(
        jnp.asarray(mean), calib.apply(jnp.asarray(var)), jnp.asarray(target)
    )
    assert after < before
    assert np.all(np.abs(np.asarray(calib.scale) - 2.0) < 0.2)


def test_metrics_basic():
    import jax.numpy as jnp

    pred = jnp.zeros((10, 3))
    target = jnp.ones((10, 3))
    assert np.allclose(np.asarray(metrics.rmse(pred, target)), 1.0)
    assert np.allclose(np.asarray(metrics.mae(pred, target)), 1.0)


def test_picp_well_calibrated():
    import jax.numpy as jnp

    rng = np.random.default_rng(1)
    n = 20000
    mean = jnp.zeros((n, 1))
    var = jnp.ones((n, 1))
    target = jnp.asarray(rng.normal(size=(n, 1)))
    cov68 = float(metrics.picp(mean, var, target, 0.68, reduce=True))
    assert abs(cov68 - 0.68) < 0.02
