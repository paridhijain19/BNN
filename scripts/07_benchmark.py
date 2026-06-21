"""Stage 07: benchmark LINX (or mock) vs. the trained emulator (speed + accuracy)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

from _common import load_config, load_or_generate_bundle, train_core
from benchmark_emulator import run_benchmark
from bbnjax.models import make_physical
from bbnjax.simulator import get_simulator
from bbnjax.tracking import ExperimentTracker
from bbnjax.viz import plots, set_style


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train_mlp.yaml")
    parser.add_argument("--out", default="artifacts/benchmark")
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_style()

    bundle = load_or_generate_bundle(cfg)
    tracker = ExperimentTracker(args.out, config=dict(cfg), seed=int(cfg.get("seed", 0)))

    core, _ = train_core("residual", bundle, cfg["model"], cfg["train"], seed=int(cfg.get("seed", 0)))
    emulator = make_physical(core, bundle)
    sim = get_simulator(cfg["simulator"]["backend"])

    result = run_benchmark(sim, emulator, bundle.space)
    print("[benchmark]")
    for n, s, e, sp in zip(result["sizes"], result["sim_times"], result["emu_times"], result["speedup"]):
        print(f"   N={n:5d}  sim={s:.4f}s  emu={e:.4f}s  speedup={sp:.1f}x")
    print("   accuracy:", result["accuracy"])
    tracker.log_summary(result)

    plots.plot_benchmark(
        result["sizes"], result["sim_times"], result["emu_times"], save=tracker.path("benchmark.png")
    )
    print(f"[done] benchmark written to {args.out}")


if __name__ == "__main__":
    main()
