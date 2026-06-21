r"""Analytic *mock* BBN simulator (smooth, differentiable, JAX-native).

This is **not** a physically accurate BBN code. It is a deliberately simple,
smooth surrogate built from published-style *linearized log-derivatives* around
the fiducial point, with mild nonlinear cross terms so the emulator has
something non-trivial to learn. Its purpose is threefold:

1. let the *entire* BBN-JAX pipeline run and be unit-tested without LINX;
2. provide a known, perfectly-differentiable ground truth for validating the
   emulator, the autodiff sensitivities, and the active-learning machinery;
3. serve as a fast smoke-test target in CI.

Each abundance is modeled as

.. math::
    A(\mathbf c) = A_0 \, \exp\!\Big( \sum_k b^{(A)}_k\, x_k
                    + \tfrac12 \sum_{kl} q^{(A)}_{kl}\, x_k x_l \Big),

with normalized deviations

.. math::
    x_b=\ln(\Omega_b h^2/\Omega_{b,0}h^2),\quad
    x_N=N_{\rm eff,tot}-3.044,\quad
    x_\tau=(\tau_n-878.4)/878.4,\quad
    x_\xi=\xi_e,

where :math:`N_{\rm eff,tot}=N_{\rm eff}+\Delta N_{\rm dark}`. The linear
coefficients :math:`b` reproduce, to first order, the rough magnitude and sign
of literature sensitivities (Pitrou et al. 2018; Fields et al. review). Replace
this backend with :class:`LinxSimulator` for any scientific result.
"""

from __future__ import annotations

from typing import Mapping

import jax.numpy as jnp
from jax import Array

from bbnjax.simulator.base import BBNSimulator

# Fiducial abundances (order-of-magnitude correct, illustrative only).
_A0 = {
    "Yp": 0.24700,
    "DH": 2.530e-5,
    "He3H": 1.040e-5,
    "Li7H": 4.700e-10,
}

# Linear log-derivatives b_k for [x_b, x_N, x_tau, x_xi]  (illustrative).
_B = {
    "Yp": jnp.asarray([0.039, 0.0660, 0.729, -1.00]),
    "DH": jnp.asarray([-1.650, 0.060, 0.400, 0.30]),
    "He3H": jnp.asarray([-0.570, 0.020, 0.140, 0.10]),
    "Li7H": jnp.asarray([2.110, -0.070, 0.400, -0.50]),
}

# Mild symmetric quadratic cross terms q_kl (nonlinearity for the emulator).
_Q = {
    "Yp": jnp.asarray(
        [
            [0.00, 0.00, 0.00, 0.00],
            [0.00, 0.00, 0.05, 0.00],
            [0.00, 0.05, 0.00, 0.00],
            [0.00, 0.00, 0.00, 0.10],
        ]
    ),
    "DH": jnp.asarray(
        [
            [0.20, -0.05, 0.00, 0.00],
            [-0.05, 0.00, 0.00, 0.00],
            [0.00, 0.00, 0.00, 0.00],
            [0.00, 0.00, 0.00, 0.00],
        ]
    ),
    "He3H": jnp.asarray(
        [
            [0.05, 0.00, 0.00, 0.00],
            [0.00, 0.00, 0.00, 0.00],
            [0.00, 0.00, 0.00, 0.00],
            [0.00, 0.00, 0.00, 0.00],
        ]
    ),
    "Li7H": jnp.asarray(
        [
            [0.30, 0.00, 0.00, 0.00],
            [0.00, 0.00, 0.00, 0.00],
            [0.00, 0.00, 0.00, 0.00],
            [0.00, 0.00, 0.00, 0.00],
        ]
    ),
}

_OMEGA_B0 = 0.02237
_TAU0 = 878.4
_NEFF0 = 3.044


class MockBBNSimulator(BBNSimulator):
    """Smooth analytic stand-in for a real BBN code (see module docstring)."""

    supports_vmap = True
    name = "mock"

    def _predict_named(self, params: Mapping[str, Array]) -> dict[str, Array]:
        omega_b = jnp.asarray(params["omega_b"])
        n_eff = jnp.asarray(params["N_eff"])
        tau_n = jnp.asarray(params["tau_n"])
        xi_e = jnp.asarray(params["xi_e"])
        dn_dark = jnp.asarray(params.get("dNeff_dark", 0.0))

        x_b = jnp.log(omega_b / _OMEGA_B0)
        x_n = (n_eff + dn_dark) - _NEFF0
        x_t = (tau_n - _TAU0) / _TAU0
        x_x = xi_e
        x = jnp.stack([x_b, x_n, x_t, x_x])

        out: dict[str, Array] = {}
        for key in ("Yp", "DH", "He3H", "Li7H"):
            lin = jnp.dot(_B[key], x)
            quad = 0.5 * jnp.dot(x, jnp.dot(_Q[key], x))
            out[key] = _A0[key] * jnp.exp(lin + quad)
        # Yp is a mass fraction; keep it physical.
        out["Yp"] = jnp.clip(out["Yp"], 0.0, 1.0)
        return out
