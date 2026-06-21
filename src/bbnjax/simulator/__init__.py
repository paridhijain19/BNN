"""Ground-truth BBN simulators (LINX) and a differentiable analytic mock."""

from __future__ import annotations

from bbnjax.simulator.base import BBNSimulator
from bbnjax.simulator.mock import MockBBNSimulator

__all__ = ["BBNSimulator", "MockBBNSimulator", "get_simulator"]


def get_simulator(backend: str = "auto", **kwargs) -> BBNSimulator:
    """Factory for BBN simulators.

    Parameters
    ----------
    backend:
        * ``"linx"`` -- require the real LINX backend (raises if unavailable).
        * ``"mock"`` -- the analytic differentiable mock.
        * ``"auto"`` -- LINX if importable, else mock (prints a notice).
    **kwargs:
        Forwarded to the backend constructor (e.g. ``network=...``).
    """
    backend = backend.lower()
    if backend == "mock":
        return MockBBNSimulator()
    if backend == "linx":
        from bbnjax.simulator.linx_adapter import LinxSimulator

        return LinxSimulator(**kwargs)
    if backend == "auto":
        try:
            from bbnjax.simulator.linx_adapter import LinxSimulator

            return LinxSimulator(**kwargs)
        except Exception:
            import warnings

            warnings.warn(
                "LINX not available; falling back to the analytic mock simulator. "
                "Results are illustrative only. Install LINX for science runs.",
                stacklevel=2,
            )
            return MockBBNSimulator()
    raise ValueError(f"Unknown simulator backend: {backend!r}")
