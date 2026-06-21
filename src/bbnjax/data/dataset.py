"""Dataset container for BBN emulation.

Holds physical-unit inputs ``C`` (cosmological parameters) and outputs ``Y``
(abundances) for train/val/test splits, together with the fitted input/target
scalers needed to move between physical and model space.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from jax import Array

from bbnjax.data.scaler import Standardizer, TargetTransform
from bbnjax.parameters import TARGETS, ParameterSpace
from bbnjax.utils.io import load_npz, save_npz

_SPLITS = ("train", "val", "test")


@dataclass
class DataBundle:
    """Train/val/test BBN data plus fitted scalers (scalers fit on *train* only)."""

    space: ParameterSpace
    target_names: tuple[str, ...]
    log_mask: np.ndarray
    data: dict[str, dict[str, np.ndarray]]  # split -> {"C": (n,D), "Y": (n,T)}
    input_scaler: Standardizer
    target_transform: TargetTransform

    # -- construction -----------------------------------------------------------------

    @classmethod
    def build(
        cls,
        space: ParameterSpace,
        splits: dict[str, dict[str, np.ndarray]],
        target_names: tuple[str, ...] = TARGETS,
        log_targets: tuple[str, ...] = ("DH", "He3H", "Li7H"),
    ) -> "DataBundle":
        log_mask = np.array([name in log_targets for name in target_names], dtype=bool)
        c_train = jnp.asarray(splits["train"]["C"])
        y_train = jnp.asarray(splits["train"]["Y"])
        input_scaler = Standardizer.fit(c_train)
        target_transform = TargetTransform.fit(y_train, jnp.asarray(log_mask))
        data = {
            s: {"C": np.asarray(splits[s]["C"]), "Y": np.asarray(splits[s]["Y"])}
            for s in splits
        }
        return cls(
            space=space,
            target_names=tuple(target_names),
            log_mask=log_mask,
            data=data,
            input_scaler=input_scaler,
            target_transform=target_transform,
        )

    # -- accessors --------------------------------------------------------------------

    @property
    def input_dim(self) -> int:
        return self.space.dim

    @property
    def output_dim(self) -> int:
        return len(self.target_names)

    def raw(self, split: str) -> tuple[Array, Array]:
        """Physical-unit ``(C, Y)`` for a split."""
        d = self.data[split]
        return jnp.asarray(d["C"]), jnp.asarray(d["Y"])

    def xy(self, split: str) -> tuple[Array, Array]:
        """Model-space ``(X, T)`` = standardized inputs / transformed targets."""
        c, y = self.raw(split)
        return self.input_scaler.forward(c), self.target_transform.forward(y)

    def n(self, split: str) -> int:
        return int(self.data[split]["C"].shape[0])

    # -- persistence ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        payload: dict[str, np.ndarray] = {
            "param_names": np.array(self.space.names),
            "param_lo": np.asarray(self.space.lo()),
            "param_hi": np.asarray(self.space.hi()),
            "param_fid": np.asarray(self.space.fiducial_vector()),
            "target_names": np.array(self.target_names),
            "log_mask": self.log_mask,
        }
        for s in self.data:
            payload[f"{s}_C"] = self.data[s]["C"]
            payload[f"{s}_Y"] = self.data[s]["Y"]
        save_npz(path, **payload)

    @classmethod
    def load(cls, path: str | Path) -> "DataBundle":
        raw = load_npz(path)
        names = tuple(str(x) for x in raw["param_names"])
        lo = raw["param_lo"]
        hi = raw["param_hi"]
        fid = raw["param_fid"]
        space = ParameterSpace(
            names=names,
            bounds={n: (float(lo[i]), float(hi[i])) for i, n in enumerate(names)},
            fiducial={n: float(fid[i]) for i, n in enumerate(names)},
        )
        target_names = tuple(str(x) for x in raw["target_names"])
        log_targets = tuple(t for t, m in zip(target_names, raw["log_mask"]) if m)
        splits = {}
        for s in _SPLITS:
            if f"{s}_C" in raw:
                splits[s] = {"C": raw[f"{s}_C"], "Y": raw[f"{s}_Y"]}
        return cls.build(space, splits, target_names=target_names, log_targets=log_targets)
