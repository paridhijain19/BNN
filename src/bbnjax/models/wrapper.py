r"""Physical-space emulator wrapper.

Core models operate in *standardized model space*. For science (sensitivities,
Fisher, likelihoods) we want a differentiable map directly in physical units:

.. math:: c\ (\text{cosmological params}) \;\longmapsto\; Y\ (\text{abundances}).

:class:`PhysicalEmulator` composes ``input_scaler``  ->  ``core``  ->
``target_transform.inverse`` into one ``eqx.Module`` that is differentiable in
``c`` end to end. It also surfaces predictive uncertainty when the core supports
it (NN+GP, deep ensemble).
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jax import Array

from bbnjax.data.scaler import Standardizer, TargetTransform


def core_predict(core: eqx.Module, x: Array) -> tuple[Array, Array]:
    """Return ``(mean, variance)`` in transformed/standardized target space.

    Falls back to zero variance for deterministic cores (plain MLP / residual).
    """
    if hasattr(core, "predict"):
        return core.predict(x)
    mean = core(x)
    return mean, jnp.zeros_like(mean)


class PhysicalEmulator(eqx.Module):
    """End-to-end differentiable emulator in physical units."""

    core: eqx.Module
    input_scaler: Standardizer
    target_transform: TargetTransform

    def __call__(self, c: Array) -> Array:
        """Physical abundances ``(T,)`` for physical parameters ``c`` ``(D,)``."""
        x = self.input_scaler.forward(c)
        t = self.core(x)
        return self.target_transform.inverse(t)

    def predict_transformed(self, c: Array) -> tuple[Array, Array]:
        """``(mean, variance)`` in the standardized target space (UQ-native space)."""
        x = self.input_scaler.forward(c)
        return core_predict(self.core, x)

    def predict(self, c: Array) -> tuple[Array, Array]:
        """Physical ``(mean, std)``.

        Uncertainty is propagated by linearizing the (per-target) inverse
        transform around the predicted mean: for linear targets the std simply
        rescales by the standardizer's ``std``; for ``log10`` targets it maps to
        a multiplicative (log-normal) spread, returned here as the 1-sigma
        absolute width ``Y * ln(10) * sigma_log``.
        """
        mean_t, var_t = self.predict_transformed(c)
        mean_phys = self.target_transform.inverse(mean_t)
        std_t = jnp.sqrt(jnp.clip(var_t, 0.0, None))
        # undo standardizer scaling -> std in (log10 or linear) target units
        std_unit = std_t * self.target_transform.standardizer.std
        log_mask = self.target_transform.log_mask
        std_phys = jnp.where(
            log_mask,
            mean_phys * jnp.log(10.0) * std_unit,  # delta-method for 10**(.)
            std_unit,
        )
        return mean_phys, std_phys


def make_physical(core: eqx.Module, bundle) -> PhysicalEmulator:
    """Wrap a trained core with a :class:`DataBundle`'s fitted scalers."""
    return PhysicalEmulator(
        core=core,
        input_scaler=bundle.input_scaler,
        target_transform=bundle.target_transform,
    )
