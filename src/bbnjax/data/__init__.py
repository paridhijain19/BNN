"""Dataset generation, containers, and scalers."""

from __future__ import annotations

from bbnjax.data.dataset import DataBundle
from bbnjax.data.generate import generate_dataset, generate_split, sample_inputs
from bbnjax.data.scaler import Standardizer, TargetTransform

__all__ = [
    "DataBundle",
    "Standardizer",
    "TargetTransform",
    "generate_dataset",
    "generate_split",
    "sample_inputs",
]
