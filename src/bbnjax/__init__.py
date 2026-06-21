"""BBN-JAX: differentiable, uncertainty-aware, active-learning ML emulators for Big Bang Nucleosynthesis.

Public API is intentionally small and import-light. Heavy submodules (training, viz, inference)
are imported lazily by user code to keep ``import bbnjax`` fast.
"""

from __future__ import annotations

__version__ = "0.1.0"

# Enable double precision globally. BBN abundances (esp. D/H, 7Li/H) span many
# orders of magnitude and Fisher matrices can be ill-conditioned, so float64 is
# the safe default for a scientific package. Must run before any JAX array is made.
import os as _os

if _os.environ.get("BBNJAX_DISABLE_X64", "0") != "1":
    import jax as _jax

    _jax.config.update("jax_enable_x64", True)

from bbnjax.parameters import ParameterSpace
from bbnjax.sampling import latin_hypercube, sobol_design

__all__ = [
    "__version__",
    "ParameterSpace",
    "latin_hypercube",
    "sobol_design",
]
