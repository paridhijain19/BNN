"""Shared pytest fixtures and path bootstrap."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import jax  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from bbnjax.data.dataset import DataBundle  # noqa: E402
from bbnjax.parameters import ParameterSpace  # noqa: E402
from bbnjax.sampling import make_design  # noqa: E402
from bbnjax.simulator import MockBBNSimulator  # noqa: E402


@pytest.fixture(scope="session")
def space() -> ParameterSpace:
    return ParameterSpace.default()


@pytest.fixture(scope="session")
def sim() -> MockBBNSimulator:
    return MockBBNSimulator()


@pytest.fixture(scope="session")
def bundle(space, sim) -> DataBundle:
    def split(n, seed):
        C = space.from_unit(make_design("sobol", n, space.dim, seed=seed))
        Y = sim.predict_batch(C, space)
        return {"C": np.asarray(C), "Y": np.asarray(Y)}

    splits = {"train": split(192, 1), "val": split(64, 2), "test": split(64, 3)}
    return DataBundle.build(space, splits)
