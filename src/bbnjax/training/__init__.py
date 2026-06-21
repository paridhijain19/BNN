"""Training utilities: losses, schedules, and the main training loop."""

from __future__ import annotations

from bbnjax.training.losses import gaussian_nll_loss, mse_loss
from bbnjax.training.schedules import build_optimizer, warmup_cosine
from bbnjax.training.trainer import TrainResult, train_ensemble, train_model

__all__ = [
    "mse_loss",
    "gaussian_nll_loss",
    "build_optimizer",
    "warmup_cosine",
    "train_model",
    "train_ensemble",
    "TrainResult",
]
