"""Abstract ground-truth BBN simulator interface.

A simulator maps cosmological parameters to the four primordial abundances in
**linear units**:

* ``Yp``   -- helium-4 mass fraction (dimensionless)
* ``DH``   -- deuterium ratio D/H
* ``He3H`` -- helium-3 ratio 3He/H
* ``Li7H`` -- lithium-7 ratio 7Li/H

Concrete backends implement :meth:`_predict_named`, which receives a *complete*
parameter dictionary (active parameters overridden, the rest at fiducial). The
public :meth:`predict` / :meth:`predict_batch` handle the mapping from an active
parameter vector (ordered by a :class:`ParameterSpace`) to that dictionary and
stack outputs into the canonical target order.
"""

from __future__ import annotations

import abc
from typing import Mapping

import jax
import jax.numpy as jnp
from jax import Array

from bbnjax.parameters import DEFAULT_FIDUCIAL, TARGETS, ParameterSpace


class BBNSimulator(abc.ABC):
    """Base class for BBN ground-truth simulators."""

    #: Whether :meth:`_predict_named` is pure-JAX and safe to ``vmap``.
    supports_vmap: bool = True

    #: Backend name for logging / provenance.
    name: str = "base"

    @abc.abstractmethod
    def _predict_named(self, params: Mapping[str, Array]) -> dict[str, Array]:
        """Return abundances for a *complete* parameter dict (linear units)."""

    # -- public API -------------------------------------------------------------------

    def predict_dict(self, params: Mapping[str, float | Array]) -> dict[str, Array]:
        """Predict from a (possibly partial) ``name -> value`` dict.

        Missing parameters are filled from :data:`DEFAULT_FIDUCIAL`.
        """
        full = {k: jnp.asarray(v) for k, v in DEFAULT_FIDUCIAL.items()}
        for k, v in params.items():
            full[k] = jnp.asarray(v)
        return self._predict_named(full)

    def predict(self, c: Array, space: ParameterSpace) -> Array:
        """Predict abundances for one active-parameter vector ``c``.

        Returns a ``(4,)`` array in canonical target order ``(Yp, DH, He3H, Li7H)``.
        Differentiable in ``c`` when the backend is pure-JAX.
        """
        c = jnp.asarray(c)
        full = dict(DEFAULT_FIDUCIAL)
        full = {k: jnp.asarray(v) for k, v in full.items()}
        for i, n in enumerate(space.names):
            full[n] = c[i]
        out = self._predict_named(full)
        return jnp.stack([out[t] for t in TARGETS])

    def predict_batch(self, C: Array, space: ParameterSpace) -> Array:
        """Predict for a batch of active-parameter vectors ``C`` of shape ``(N, D)``.

        Returns ``(N, 4)``. Uses ``vmap`` when the backend supports it, otherwise
        falls back to a Python loop (e.g. for non-traceable external simulators).
        """
        C = jnp.atleast_2d(jnp.asarray(C))
        if self.supports_vmap:
            return jax.vmap(lambda row: self.predict(row, space))(C)
        rows = [self.predict(C[i], space) for i in range(C.shape[0])]
        return jnp.stack(rows)
