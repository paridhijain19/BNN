"""Generate BBN training datasets from a simulator using space-filling designs."""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from bbnjax.config import load_config
from bbnjax.data.dataset import DataBundle
from bbnjax.parameters import ParameterSpace
from bbnjax.sampling import make_design
from bbnjax.simulator import BBNSimulator, get_simulator


def sample_inputs(
    space: ParameterSpace,
    n: int,
    design: str = "sobol",
    *,
    seed: int = 0,
    scramble: bool = True,
    key: Array | None = None,
) -> Array:
    """Draw ``n`` physical-unit parameter vectors with the given design."""
    if key is None:
        key = jax.random.PRNGKey(seed)
    u = make_design(design, n, space.dim, key=key, seed=seed, scramble=scramble)
    return space.from_unit(u)


def generate_split(
    sim: BBNSimulator,
    space: ParameterSpace,
    n: int,
    design: str = "sobol",
    *,
    seed: int = 0,
    scramble: bool = True,
    key: Array | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate one ``(C, Y)`` split of physical inputs/outputs."""
    c = sample_inputs(space, n, design, seed=seed, scramble=scramble, key=key)
    y = sim.predict_batch(c, space)
    return np.asarray(c), np.asarray(y)


def generate_dataset(cfg) -> DataBundle:
    """Build and persist a :class:`DataBundle` from a parsed config."""
    space = ParameterSpace.from_config(cfg["parameters"])
    sim = get_simulator(cfg["simulator"]["backend"])
    data_cfg = cfg["data"]
    base_seed = int(cfg.get("seed", 0))
    design = data_cfg.get("design", "sobol")
    scramble = bool(data_cfg.get("scramble", True))

    keys = jax.random.split(jax.random.PRNGKey(base_seed), 3)
    splits = {}
    for i, (split, n_key) in enumerate(
        zip(("train", "val", "test"), ("n_train", "n_val", "n_test"))
    ):
        n = int(data_cfg[n_key])
        if n <= 0:
            continue
        c, y = generate_split(
            sim, space, n, design, seed=base_seed + 1 + i, scramble=scramble, key=keys[i]
        )
        splits[split] = {"C": c, "Y": y}

    bundle = DataBundle.build(
        space,
        splits,
        target_names=tuple(cfg["targets"]),
        log_targets=tuple(cfg.get("log_targets", ("DH", "He3H", "Li7H"))),
    )

    out_dir = Path(data_cfg.get("out_dir", "data/processed"))
    name = data_cfg.get("name", "bbn_dataset")
    out_path = out_dir / f"{name}.npz"
    bundle.save(out_path)
    print(
        f"[generate] backend={sim.name} design={design} "
        f"train={bundle.n('train')} val={bundle.n('val') if 'val' in bundle.data else 0} "
        f"test={bundle.n('test') if 'test' in bundle.data else 0} -> {out_path}"
    )
    return bundle


def cli() -> None:
    parser = argparse.ArgumentParser(description="Generate a BBN emulator dataset.")
    parser.add_argument("--config", required=True, help="Path to data_gen YAML config.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    generate_dataset(cfg)


if __name__ == "__main__":  # pragma: no cover
    cli()
