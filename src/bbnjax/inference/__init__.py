"""Differentiable inference: sensitivities, Fisher forecasts, likelihoods."""

from __future__ import annotations

from bbnjax.inference.fisher import (
    ObservationalData,
    fisher_matrix,
    forecast_report,
    marginal_errors,
    parameter_covariance,
)
from bbnjax.inference.likelihood import BBNLikelihood, make_cobaya_likelihood
from bbnjax.inference.sensitivity import (
    abundance_jacobian,
    batched_jacobian,
    log_sensitivity,
)

__all__ = [
    "abundance_jacobian",
    "log_sensitivity",
    "batched_jacobian",
    "ObservationalData",
    "fisher_matrix",
    "parameter_covariance",
    "marginal_errors",
    "forecast_report",
    "BBNLikelihood",
    "make_cobaya_likelihood",
]
