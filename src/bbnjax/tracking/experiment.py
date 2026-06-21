"""Lightweight, dependency-free experiment tracking and model persistence.

Each run gets a directory containing:

* ``manifest.json`` -- provenance (git SHA, timestamp, JAX/NumPy versions, seed,
  config snapshot) for reproducibility;
* ``metrics.jsonl`` -- one JSON record per logged step;
* ``summary.json``  -- final scalar metrics;
* artifacts (pickled emulators, figures).

This keeps the package self-contained; swap in W&B/MLflow by subclassing
:class:`ExperimentTracker` if desired.
"""

from __future__ import annotations

import datetime as _dt
import pickle
import platform
from pathlib import Path
from typing import Any

from bbnjax.utils.io import ensure_dir, git_revision, save_json


class ExperimentTracker:
    """Minimal run logger writing a manifest, metric stream, and summary."""

    def __init__(self, run_dir: str | Path, config: dict | None = None, seed: int | None = None):
        self.run_dir = ensure_dir(run_dir)
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self._records: list[dict] = []
        self._write_manifest(config or {}, seed)
        if config is not None:
            save_json(dict(config), self.run_dir / "config.json")

    def _write_manifest(self, config: dict, seed: int | None) -> None:
        import jax
        import numpy as np

        manifest = {
            "created": _dt.datetime.now().isoformat(timespec="seconds"),
            "git_sha": git_revision(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "jax": jax.__version__,
            "numpy": np.__version__,
            "x64": bool(jax.config.read("jax_enable_x64")),
            "seed": seed,
        }
        save_json(manifest, self.run_dir / "manifest.json")

    def log_metrics(self, step: int, metrics: dict[str, Any]) -> None:
        record = {"step": int(step), **{k: _to_native(v) for k, v in metrics.items()}}
        self._records.append(record)
        with open(self.metrics_path, "a", encoding="utf-8") as fh:
            import json

            fh.write(json.dumps(record) + "\n")

    def log_summary(self, summary: dict[str, Any]) -> None:
        save_json(summary, self.run_dir / "summary.json")

    def save_artifact(self, name: str, obj: Any) -> Path:
        path = self.run_dir / name
        with open(path, "wb") as fh:
            pickle.dump(obj, fh)
        return path

    def path(self, *parts: str) -> Path:
        return self.run_dir.joinpath(*parts)


def _to_native(v: Any) -> Any:
    try:
        import numpy as np

        if isinstance(v, (np.generic,)):
            return v.item()
        if hasattr(v, "tolist") and getattr(v, "ndim", 1) == 0:
            return v.tolist()
    except Exception:
        pass
    return v


# -- emulator persistence ------------------------------------------------------------


def save_emulator(path: str | Path, emulator, space) -> Path:
    """Persist a (physical) emulator + its parameter space via pickle.

    Equinox modules are dataclasses of arrays and pickle cleanly provided the
    defining classes are importable. For long-term/array-only storage consider
    ``eqx.tree_serialise_leaves`` alongside a stored architecture spec.
    """
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "wb") as fh:
        pickle.dump({"emulator": emulator, "space": space}, fh)
    return path


def load_emulator(path: str | Path):
    """Load ``(space, emulator)`` saved by :func:`save_emulator`."""
    with open(path, "rb") as fh:
        payload = pickle.load(fh)
    return payload["space"], payload["emulator"]
