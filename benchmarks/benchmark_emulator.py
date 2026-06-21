"""Benchmark suite: LINX (or mock) ground truth vs. the emulator.

Measures (i) throughput (wall-time vs. number of evaluations) and (ii) accuracy
(emulator error against the simulator on a fresh design). Designed to produce the
speed/accuracy trade-off numbers reported in the paper.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from bbnjax.parameters import ParameterSpace  # noqa: E402
from bbnjax.sampling import make_design  # noqa: E402
from bbnjax.uq import metrics  # noqa: E402


def _time_call(fn, *args, repeats: int = 3) -> float:
    fn(*args)  # warmup / compile
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn(*args)
        jax.block_until_ready(out)
        best = min(best, time.perf_counter() - t0)
    return best


def throughput(sim, emulator_batch, space: ParameterSpace, sizes=(16, 64, 256, 1024)):
    """Return ``(sizes, sim_times, emu_times)`` wall-times in seconds."""
    sim_times, emu_times = [], []
    for n in sizes:
        C = space.from_unit(make_design("sobol", n, space.dim, seed=n))
        sim_times.append(_time_call(lambda c=C: sim.predict_batch(c, space)))
        emu_times.append(_time_call(lambda c=C: emulator_batch(c)))
    return list(sizes), sim_times, emu_times


def accuracy(sim, emulator, space: ParameterSpace, n: int = 512) -> dict:
    """Relative-error metrics of the emulator vs. the simulator on a fresh design."""
    C = space.from_unit(make_design("sobol", n, space.dim, seed=99_999))
    Y_true = np.asarray(sim.predict_batch(C, space))
    Y_emu = np.asarray(jax.vmap(emulator)(C))
    rel = np.abs(Y_emu - Y_true) / (np.abs(Y_true) + 1e-30)
    return {
        "median_rel_error": [float(v) for v in np.median(rel, axis=0)],
        "max_rel_error": [float(v) for v in np.max(rel, axis=0)],
        "rmse": [float(v) for v in metrics.rmse(jnp.asarray(Y_emu), jnp.asarray(Y_true))],
    }


def run_benchmark(sim, emulator, space: ParameterSpace, sizes=(16, 64, 256, 1024)) -> dict:
    """Full benchmark: throughput + accuracy + speedup factors."""
    emu_batch = jax.jit(lambda C: jax.vmap(emulator)(C))
    sizes, sim_t, emu_t = throughput(sim, emu_batch, space, sizes=sizes)
    speedup = [s / e if e > 0 else float("inf") for s, e in zip(sim_t, emu_t)]
    return {
        "backend": sim.name,
        "sizes": sizes,
        "sim_times": sim_t,
        "emu_times": emu_t,
        "speedup": speedup,
        "accuracy": accuracy(sim, emulator, space),
    }


if __name__ == "__main__":  # pragma: no cover
    from bbnjax.simulator import get_simulator

    space = ParameterSpace.default()
    sim = get_simulator("auto")
    # Identity-ish emulator placeholder when run standalone.
    fid = sim.predict(space.fiducial_vector(), space)
    emulator = lambda c: fid
    print(run_benchmark(sim, emulator, space))
