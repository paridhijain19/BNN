from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from bbnjax.active_learning.acquisition import score_bald, score_max_variance, select_batch
from bbnjax.active_learning.loop import run_active_learning


def test_acquisition_scores():
    var = jnp.array([[0.1, 0.2], [0.5, 0.5], [0.0, 0.0]])
    s = np.asarray(score_max_variance(var))
    assert np.argmax(s) == 1
    bald = np.asarray(score_bald(var + 0.1, jnp.full_like(var, 0.1)))
    assert np.all(bald >= 0.0)


def test_select_batch_topk():
    scores = jnp.array([0.1, 0.9, 0.5, 0.7])
    idx = np.asarray(select_batch(scores, 2))
    assert set(idx.tolist()) == {1, 3}


def _cfg():
    return {
        "seed": 0,
        "parameters": {
            "active": ["omega_b", "N_eff"],
            "bounds": {"omega_b": [0.020, 0.024], "N_eff": [2.5, 3.5]},
            "fiducial": {"omega_b": 0.02237, "N_eff": 3.044},
        },
        "targets": ["Yp", "DH", "He3H", "Li7H"],
        "log_targets": ["DH", "He3H", "Li7H"],
        "simulator": {"backend": "mock"},
        "active_learning": {
            "init_size": 48,
            "rounds": 2,
            "batch_per_round": 16,
            "pool_size": 256,
            "acquisition": "max_variance",
            "retrain_epochs": 30,
        },
        "model": {"width": 16, "depth": 2, "n_members": 2},
        "train": {"batch_size": 32, "lr": 3e-3, "ema_decay": 0.999},
    }


def test_active_learning_runs():
    history, bundle, ensemble = run_active_learning(_cfg())
    h = history.as_dict()
    assert len(h["n_train"]) == 3  # rounds + 1 evaluations
    assert h["n_train"][0] == 48
    assert h["n_train"][-1] == 48 + 2 * 16
    assert all(np.isfinite(h["test_rmse_mean"]))
