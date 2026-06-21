"""Shared helpers and path bootstrap for the pipeline scripts.

Scripts can be run directly (``python scripts/01_train_mlp.py``) without
installing the package; this module puts ``src/`` on ``sys.path``.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import jax  # noqa: E402

from bbnjax.config import load_config  # noqa: E402
from bbnjax.data.dataset import DataBundle  # noqa: E402
from bbnjax.data.generate import generate_dataset  # noqa: E402
from bbnjax.models import build_model  # noqa: E402
from bbnjax.training import train_model  # noqa: E402


def load_or_generate_bundle(cfg) -> DataBundle:
    """Load the dataset referenced by ``cfg['data']['path']`` or generate a small one."""
    data_cfg = cfg.get("data", {})
    path = data_cfg.get("path")
    if path and Path(path).exists():
        print(f"[data] loading {path}")
        return DataBundle.load(path)

    print("[data] no dataset found; generating a small mock dataset")
    gen_cfg = dict(cfg)
    gen_cfg["data"] = {
        "n_train": data_cfg.get("n_train", 1024),
        "n_val": data_cfg.get("n_val", 128),
        "n_test": data_cfg.get("n_test", 256),
        "design": data_cfg.get("design", "sobol"),
        "scramble": True,
        "out_dir": "data/processed",
        "name": "bbn_auto",
    }
    return generate_dataset(gen_cfg)


def train_core(kind: str, bundle: DataBundle, model_cfg, train_cfg, *, seed: int = 0):
    """Build and train a deterministic core (mlp/residual); returns (core, result)."""
    X, T = bundle.xy("train")
    X_val, T_val = bundle.xy("val") if "val" in bundle.data else (None, None)
    key = jax.random.PRNGKey(seed)
    k_model, k_train = jax.random.split(key)
    core = build_model(
        kind, bundle.input_dim, bundle.output_dim, model_cfg, key=k_model, X_train=X, T_train=T
    )
    result = train_model(
        core,
        X,
        T,
        X_val=X_val,
        T_val=T_val,
        epochs=int(train_cfg.get("epochs", 300)),
        batch_size=int(train_cfg.get("batch_size", 256)),
        lr=float(train_cfg.get("lr", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-5)),
        grad_clip=float(train_cfg.get("grad_clip", 1.0)),
        warmup_frac=float(train_cfg.get("warmup_frac", 0.05)),
        ema_decay=train_cfg.get("ema_decay", 0.999),
        patience=train_cfg.get("patience"),
        key=k_train,
        verbose=True,
    )
    return result.model, result


__all__ = ["ROOT", "SRC", "load_config", "load_or_generate_bundle", "train_core"]
