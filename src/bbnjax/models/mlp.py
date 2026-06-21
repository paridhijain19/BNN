"""Equinox MLP emulator (optionally heteroscedastic).

Models operate in *standardized model space*: input ``x`` of shape ``(D,)`` and
output of shape ``(T,)`` (the mean), with batches handled by ``jax.vmap`` /
``eqx.filter_vmap``. A heteroscedastic variant additionally predicts a
per-target log-variance for aleatoric uncertainty (used by deep ensembles).
"""

from __future__ import annotations

from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

ACTIVATIONS: dict[str, Callable[[Array], Array]] = {
    "relu": jax.nn.relu,
    "gelu": jax.nn.gelu,
    "tanh": jax.nn.tanh,
    "silu": jax.nn.silu,
    "swish": jax.nn.silu,
    "softplus": jax.nn.softplus,
    "elu": jax.nn.elu,
}


def get_activation(name: str) -> Callable[[Array], Array]:
    try:
        return ACTIVATIONS[name.lower()]
    except KeyError as exc:  # pragma: no cover
        raise ValueError(f"Unknown activation {name!r}; choose from {list(ACTIVATIONS)}") from exc


class MLPEmulator(eqx.Module):
    """Plain or heteroscedastic MLP mapping ``(D,) -> (T,)`` in standardized space."""

    net: eqx.nn.MLP
    out_dim: int = eqx.field(static=True)
    heteroscedastic: bool = eqx.field(static=True)

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        width: int = 128,
        depth: int = 4,
        activation: str = "gelu",
        heteroscedastic: bool = False,
        key: Array,
    ) -> None:
        net_out = 2 * out_dim if heteroscedastic else out_dim
        self.net = eqx.nn.MLP(
            in_size=in_dim,
            out_size=net_out,
            width_size=width,
            depth=depth,
            activation=get_activation(activation),
            key=key,
        )
        self.out_dim = out_dim
        self.heteroscedastic = heteroscedastic

    def __call__(self, x: Array) -> Array:
        """Return the predictive mean ``(T,)`` for a single input ``x``."""
        out = self.net(x)
        if self.heteroscedastic:
            return out[: self.out_dim]
        return out

    def mean_and_logvar(self, x: Array) -> tuple[Array, Array]:
        """Return ``(mean, log_variance)`` (log-var is zeros if homoscedastic)."""
        out = self.net(x)
        if self.heteroscedastic:
            mean = out[: self.out_dim]
            log_var = out[self.out_dim :]
            # Soft floor on variance for numerical stability.
            log_var = jax.nn.softplus(log_var) - 6.0
            return mean, log_var
        return out, jnp.full((self.out_dim,), -jnp.inf)


def build_mlp(in_dim: int, out_dim: int, cfg, key: Array) -> MLPEmulator:
    """Construct an :class:`MLPEmulator` from a model config block."""
    return MLPEmulator(
        in_dim,
        out_dim,
        width=int(cfg.get("width", 128)),
        depth=int(cfg.get("depth", 4)),
        activation=cfg.get("activation", "gelu"),
        heteroscedastic=bool(cfg.get("heteroscedastic", False)),
        key=key,
    )
