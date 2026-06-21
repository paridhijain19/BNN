"""Emulator models: MLP, residual, GP, NN+GP, deep ensemble, physical wrapper."""

from __future__ import annotations

import jax

from bbnjax.models.ensemble import DeepEnsemble
from bbnjax.models.gp import GPModel, MultiOutputGP, fit_gp
from bbnjax.models.mlp import MLPEmulator, build_mlp
from bbnjax.models.nn_gp import NNGPEmulator
from bbnjax.models.residual import LinearBaseline, ResidualEmulator, build_residual
from bbnjax.models.wrapper import PhysicalEmulator, core_predict, make_physical

__all__ = [
    "MLPEmulator",
    "build_mlp",
    "ResidualEmulator",
    "LinearBaseline",
    "build_residual",
    "GPModel",
    "MultiOutputGP",
    "fit_gp",
    "NNGPEmulator",
    "DeepEnsemble",
    "PhysicalEmulator",
    "make_physical",
    "core_predict",
    "build_model",
]


def build_model(kind: str, in_dim: int, out_dim: int, cfg, *, key, X_train=None, T_train=None):
    """Construct a *core* model (standardized space) from a config block.

    Parameters
    ----------
    kind:
        ``"mlp"``, ``"residual"``, ``"ensemble"`` (NN+GP and GP are built from a
        trained core via their own ``fit`` constructors).
    """
    kind = kind.lower()
    if kind == "mlp":
        return build_mlp(in_dim, out_dim, cfg, key)
    if kind == "residual":
        if X_train is None or T_train is None:
            raise ValueError("Residual model requires X_train/T_train to fit the baseline.")
        return build_residual(in_dim, out_dim, cfg, X_train, T_train, key)
    if kind == "ensemble":
        return DeepEnsemble.init(
            in_dim,
            out_dim,
            int(cfg.get("n_members", 5)),
            width=int(cfg.get("width", 128)),
            depth=int(cfg.get("depth", 4)),
            activation=cfg.get("activation", "gelu"),
            heteroscedastic=bool(cfg.get("heteroscedastic", True)),
            key=key,
        )
    raise ValueError(f"Unknown model kind: {kind!r}")
