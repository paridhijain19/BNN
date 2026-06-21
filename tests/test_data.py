from __future__ import annotations

import numpy as np

from bbnjax.data.dataset import DataBundle
from bbnjax.data.scaler import Standardizer, TargetTransform


def test_standardizer_roundtrip():
    import jax.numpy as jnp

    x = jnp.asarray(np.random.default_rng(0).normal(size=(50, 3)) * 5 + 2)
    sc = Standardizer.fit(x)
    z = sc.forward(x)
    assert np.allclose(np.asarray(jnp.mean(z, axis=0)), 0.0, atol=1e-6)
    assert np.allclose(np.asarray(sc.inverse(z)), np.asarray(x), atol=1e-6)


def test_target_transform_log_roundtrip():
    import jax.numpy as jnp

    y = jnp.asarray([[0.25, 2.5e-5, 1e-5, 5e-10]] * 20)
    mask = jnp.asarray([False, True, True, True])
    tt = TargetTransform.fit(y, mask)
    z = tt.forward(y)
    back = tt.inverse(z)
    assert np.allclose(np.asarray(back), np.asarray(y), rtol=1e-6)


def test_bundle_save_load(tmp_path, bundle):
    p = tmp_path / "ds.npz"
    bundle.save(p)
    loaded = DataBundle.load(p)
    assert loaded.space.names == bundle.space.names
    assert loaded.target_names == bundle.target_names
    c0, y0 = bundle.raw("train")
    c1, y1 = loaded.raw("train")
    assert np.allclose(np.asarray(c0), np.asarray(c1))
    assert np.allclose(np.asarray(y0), np.asarray(y1))
