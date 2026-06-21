"""Differentiable, PyTree-friendly affine scalers and target transforms.

These are Equinox modules so they live happily inside ``jit``/``grad`` and can be
saved alongside model weights. The :class:`TargetTransform` additionally folds in
the per-target ``log10`` transform used for the wide-dynamic-range abundances.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jax import Array

_EPS = 1e-12


class Standardizer(eqx.Module):
    """Affine standardizer ``z = (x - mean) / std`` (per feature)."""

    mean: Array
    std: Array

    @classmethod
    def fit(cls, x: Array) -> "Standardizer":
        x = jnp.asarray(x)
        mean = jnp.mean(x, axis=0)
        std = jnp.std(x, axis=0)
        std = jnp.where(std < _EPS, 1.0, std)
        return cls(mean=mean, std=std)

    def forward(self, x: Array) -> Array:
        return (x - self.mean) / self.std

    def inverse(self, z: Array) -> Array:
        return z * self.std + self.mean

    def inverse_scale(self, dz: Array) -> Array:
        """Scale a *difference / std* back to physical units (no mean shift)."""
        return dz * self.std


class TargetTransform(eqx.Module):
    """Maps physical abundances <-> standardized model space.

    Forward pipeline per target:
        physical -> (log10 if in ``log_mask``) -> standardize.
    The inverse undoes both. ``log_mask`` is a static boolean array.
    """

    standardizer: Standardizer
    log_mask: Array = eqx.field(static=False)

    @classmethod
    def fit(cls, y_phys: Array, log_mask: Array) -> "TargetTransform":
        log_mask = jnp.asarray(log_mask, dtype=bool)
        y_t = cls._apply_log(jnp.asarray(y_phys), log_mask)
        return cls(standardizer=Standardizer.fit(y_t), log_mask=log_mask)

    @staticmethod
    def _apply_log(y: Array, log_mask: Array) -> Array:
        return jnp.where(log_mask, jnp.log10(jnp.clip(y, _EPS, None)), y)

    @staticmethod
    def _apply_exp(y: Array, log_mask: Array) -> Array:
        return jnp.where(log_mask, jnp.power(10.0, y), y)

    def forward(self, y_phys: Array) -> Array:
        y_t = self._apply_log(y_phys, self.log_mask)
        return self.standardizer.forward(y_t)

    def inverse(self, z: Array) -> Array:
        y_t = self.standardizer.inverse(z)
        return self._apply_exp(y_t, self.log_mask)
