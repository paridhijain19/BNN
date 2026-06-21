r"""Adapter wrapping the real LINX BBN code as a :class:`BBNSimulator`.

This maps BBN-JAX's parameter convention onto the LINX API
(``background.BackgroundModel`` + ``abundances.AbundanceModel``) and converts
LINX's raw per-species number abundances into the four observables used here.

LINX call pattern (see Giovanetti et al. 2024, arXiv:2408.14538)::

    from linx.background import BackgroundModel
    from linx.abundances import AbundanceModel
    from linx.nuclear import NuclearRates

    bkg = BackgroundModel()
    t, a, rho_g, rho_nu, rho_NP, p_NP, Neff = bkg(jnp.asarray(delta_neff_init))

    model = AbundanceModel(NuclearRates(nuclear_net="key_PRIMAT_2023"))
    Yn, Yp, Yd, Yt, YHe3, Ya, YLi7, YBe7 = model(
        rho_g, rho_nu, rho_NP, p_NP, t_vec=t, a_vec=a,
        eta_fac=eta_fac, tau_n_fac=tau_n_fac, nuclear_rates_q=q,
    )

Parameter mapping
-----------------
* ``omega_b``      -> ``eta_fac = omega_b / linx.const.Omegabh2``
* ``tau_n``        -> ``tau_n_fac = tau_n / TAU_N_LINX_DEFAULT``
* ``N_eff``/dark   -> ``delta_neff_init = (N_eff - 3.044) + dNeff_dark``
* ``xi_e``         -> **not supported by stock SBBN LINX**; requires a
  degeneracy-aware background/weak-rate scenario. Raises unless ``xi_e == 0``
  (or ``allow_unsupported=True``).

Notes
-----
The species-to-observable conversion below uses standard definitions
(mass fraction for :math:`Y_p`, number ratios for the rest, with decays of
:math:`t\to{}^3\mathrm{He}` and :math:`^7\mathrm{Be}\to{}^7\mathrm{Li}` folded
in). **Verify these against your installed LINX version's documented output
normalization before publishing.**
"""

from __future__ import annotations

from typing import Mapping

import jax.numpy as jnp
from jax import Array

from bbnjax.simulator.base import BBNSimulator

#: LINX default neutron lifetime used to form ``tau_n_fac`` (seconds).
TAU_N_LINX_DEFAULT = 879.4

#: Mass numbers for the 8 LINX species, in output order.
_MASS_NUMBER = jnp.asarray([1.0, 1.0, 2.0, 3.0, 3.0, 4.0, 7.0, 7.0])


class LinxSimulator(BBNSimulator):
    """Ground-truth BBN simulator backed by LINX.

    Parameters
    ----------
    network:
        LINX nuclear network name (e.g. ``"key_PRIMAT_2023"``,
        ``"full_PRIMAT_2023"``). Use a full network for accurate 7Li/H.
    allow_unsupported:
        If ``True``, silently ignore unsupported parameters (e.g. ``xi_e``)
        instead of raising. Use only for exploratory runs.
    """

    supports_vmap = False  # BackgroundModel ODE solves are heavy; batch manually.
    name = "linx"

    def __init__(self, network: str = "key_PRIMAT_2023", *, allow_unsupported: bool = False) -> None:
        try:
            from linx.abundances import AbundanceModel
            from linx.background import BackgroundModel
            from linx.const import Omegabh2
            from linx.nuclear import NuclearRates
        except Exception as exc:  # pragma: no cover - depends on optional dep
            raise ImportError(
                "LINX is not installed. Install it (`pip install linx`) or use the "
                "mock backend via get_simulator('mock')."
            ) from exc

        self._network = network
        self._allow_unsupported = allow_unsupported
        self._omegabh2_ref = float(Omegabh2)
        self._bkg = BackgroundModel()
        self._abund = AbundanceModel(NuclearRates(nuclear_net=network))
        self._n_reactions = len(self._abund.nuclear_net.reactions)
        self._bkg_cache: dict[float, tuple] = {}

    # -- helpers ----------------------------------------------------------------------

    def _background(self, delta_neff: float) -> tuple:
        key = round(float(delta_neff), 8)
        if key not in self._bkg_cache:
            self._bkg_cache[key] = self._bkg(jnp.asarray(key))
        return self._bkg_cache[key]

    @staticmethod
    def _to_observables(species: Array) -> dict[str, Array]:
        """Convert the 8 LINX number abundances to the 4 BBN-JAX observables."""
        Yn, Yp, Yd, Yt, YHe3, Ya, YLi7, YBe7 = species
        total_mass = jnp.dot(_MASS_NUMBER, species)
        Yp_mass = 4.0 * Ya / total_mass
        DH = Yd / Yp
        He3H = (YHe3 + Yt) / Yp
        Li7H = (YLi7 + YBe7) / Yp
        return {"Yp": Yp_mass, "DH": DH, "He3H": He3H, "Li7H": Li7H}

    # -- core ------------------------------------------------------------------------

    def _predict_named(self, params: Mapping[str, Array]) -> dict[str, Array]:
        omega_b = float(params["omega_b"])
        n_eff = float(params["N_eff"])
        tau_n = float(params["tau_n"])
        xi_e = float(params.get("xi_e", 0.0))
        dn_dark = float(params.get("dNeff_dark", 0.0))

        if abs(xi_e) > 0 and not self._allow_unsupported:
            raise NotImplementedError(
                "Non-zero xi_e (neutrino degeneracy) is not supported by the stock "
                "SBBN LINX scenario. Provide a degeneracy-aware background or pass "
                "allow_unsupported=True to ignore xi_e."
            )

        delta_neff = (n_eff - 3.044) + dn_dark
        t_vec, a_vec, rho_g, rho_nu, rho_NP, p_NP, _ = self._background(delta_neff)

        eta_fac = jnp.asarray(omega_b / self._omegabh2_ref)
        tau_n_fac = jnp.asarray(tau_n / TAU_N_LINX_DEFAULT)
        q = jnp.zeros(self._n_reactions)

        species = self._abund(
            rho_g,
            rho_nu,
            rho_NP,
            p_NP,
            t_vec=t_vec,
            a_vec=a_vec,
            eta_fac=eta_fac,
            tau_n_fac=tau_n_fac,
            nuclear_rates_q=q,
        )
        return self._to_observables(jnp.asarray(species))
