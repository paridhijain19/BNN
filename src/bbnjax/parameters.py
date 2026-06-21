"""Cosmological parameter space for BBN emulation.

Defines the input parameters, their sampling bounds, fiducial values, and the
affine transforms between *physical* units and the *unit cube* :math:`[0,1]^D`
used for space-filling designs and (optionally) for network inputs.

All array-returning methods return ``jax.numpy`` arrays so they compose with
``jit``/``grad``. The metadata (names, bounds) is static Python data so a
``ParameterSpace`` can be used as a static argument or closed over.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import jax.numpy as jnp
from jax import Array

# --- physical constants / conversions ------------------------------------------------

#: eta10 = 10^10 * eta_b ~= ETA10_PER_OMEGAB * Omega_b h^2  (Pitrou et al. 2018).
ETA10_PER_OMEGAB: float = 273.9

#: Default fiducial values (Planck 2018 base-LCDM + PDG neutron lifetime).
DEFAULT_FIDUCIAL: dict[str, float] = {
    "omega_b": 0.02237,
    "N_eff": 3.044,
    "tau_n": 878.4,
    "xi_e": 0.0,
    "dNeff_dark": 0.0,
}

#: Default sampling bounds [low, high] for each supported parameter.
DEFAULT_BOUNDS: dict[str, tuple[float, float]] = {
    "omega_b": (0.019, 0.025),
    "N_eff": (2.0, 4.5),
    "tau_n": (875.0, 882.0),
    "xi_e": (-0.10, 0.10),
    "dNeff_dark": (0.0, 1.0),
}

#: Human-readable LaTeX labels (for plotting).
LATEX_LABELS: dict[str, str] = {
    "omega_b": r"$\Omega_b h^2$",
    "N_eff": r"$N_{\rm eff}$",
    "tau_n": r"$\tau_n\,[\mathrm{s}]$",
    "xi_e": r"$\xi_{e}$",
    "dNeff_dark": r"$\Delta N_{\rm dark}$",
}


def omega_b_to_eta10(omega_b: Array | float) -> Array:
    """Convert :math:`\\Omega_b h^2` to :math:`\\eta_{10} = 10^{10}\\eta_b`."""
    return ETA10_PER_OMEGAB * jnp.asarray(omega_b)


def eta10_to_omega_b(eta10: Array | float) -> Array:
    """Inverse of :func:`omega_b_to_eta10`."""
    return jnp.asarray(eta10) / ETA10_PER_OMEGAB


@dataclass(frozen=True)
class ParameterSpace:
    """Ordered cosmological parameter space with bounds and fiducial values.

    Parameters
    ----------
    names:
        Ordered tuple of active parameter names. The order defines the column
        order of every design / input array used throughout the package.
    bounds:
        Mapping ``name -> (low, high)`` sampling bounds (physical units).
    fiducial:
        Mapping ``name -> value`` fiducial point (physical units).
    """

    names: tuple[str, ...]
    bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    fiducial: dict[str, float] = field(default_factory=dict)

    # -- constructors -----------------------------------------------------------------

    @classmethod
    def default(cls, names: Sequence[str] = ("omega_b", "N_eff", "tau_n", "xi_e")) -> "ParameterSpace":
        names = tuple(names)
        return cls(
            names=names,
            bounds={n: DEFAULT_BOUNDS[n] for n in names},
            fiducial={n: DEFAULT_FIDUCIAL[n] for n in names},
        )

    @classmethod
    def from_config(cls, cfg: Mapping) -> "ParameterSpace":
        """Build from a parsed ``parameters`` config block."""
        names = tuple(cfg["active"])
        bounds_cfg = cfg.get("bounds", {})
        fid_cfg = cfg.get("fiducial", {})
        bounds = {n: tuple(bounds_cfg.get(n, DEFAULT_BOUNDS[n])) for n in names}
        fiducial = {n: float(fid_cfg.get(n, DEFAULT_FIDUCIAL[n])) for n in names}
        return cls(names=names, bounds=bounds, fiducial=fiducial)

    # -- sizes / labels ---------------------------------------------------------------

    @property
    def dim(self) -> int:
        return len(self.names)

    @property
    def labels(self) -> list[str]:
        return [LATEX_LABELS.get(n, n) for n in self.names]

    def index(self, name: str) -> int:
        return self.names.index(name)

    # -- arrays -----------------------------------------------------------------------

    def lo(self) -> Array:
        return jnp.asarray([self.bounds[n][0] for n in self.names], dtype=jnp.float64)

    def hi(self) -> Array:
        return jnp.asarray([self.bounds[n][1] for n in self.names], dtype=jnp.float64)

    def bounds_array(self) -> Array:
        """``(D, 2)`` array of ``[low, high]`` per parameter."""
        return jnp.stack([self.lo(), self.hi()], axis=-1)

    def fiducial_vector(self) -> Array:
        return jnp.asarray([self.fiducial[n] for n in self.names], dtype=jnp.float64)

    def width(self) -> Array:
        return self.hi() - self.lo()

    # -- transforms: physical <-> unit cube -------------------------------------------

    def to_unit(self, x: Array) -> Array:
        """Map physical parameters to :math:`[0,1]^D` (affine, per-dimension)."""
        return (jnp.asarray(x) - self.lo()) / self.width()

    def from_unit(self, u: Array) -> Array:
        """Map unit-cube points back to physical parameters."""
        return self.lo() + jnp.asarray(u) * self.width()

    def clip(self, x: Array) -> Array:
        """Clip physical parameters to their bounds."""
        return jnp.clip(jnp.asarray(x), self.lo(), self.hi())

    def contains(self, x: Array) -> Array:
        """Boolean mask of whether each row lies inside the bounds."""
        x = jnp.atleast_2d(jnp.asarray(x))
        return jnp.all((x >= self.lo()) & (x <= self.hi()), axis=-1)

    # -- convenience ------------------------------------------------------------------

    def as_dict(self, x: Array) -> dict[str, Array]:
        """Turn a single physical vector into a ``name -> value`` dict."""
        x = jnp.asarray(x)
        return {n: x[i] for i, n in enumerate(self.names)}

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        items = ", ".join(f"{n}={self.bounds[n]}" for n in self.names)
        return f"ParameterSpace({items})"


#: Canonical output (target) ordering used everywhere in the package.
TARGETS: tuple[str, ...] = ("Yp", "DH", "He3H", "Li7H")

#: Targets emulated in log10 space by default.
DEFAULT_LOG_TARGETS: tuple[str, ...] = ("DH", "He3H", "Li7H")

TARGET_LABELS: dict[str, str] = {
    "Yp": r"$Y_p$",
    "DH": r"$\mathrm{D/H}$",
    "He3H": r"$^3\mathrm{He}/\mathrm{H}$",
    "Li7H": r"$^7\mathrm{Li}/\mathrm{H}$",
}
