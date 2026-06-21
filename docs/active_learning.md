# Active-learning strategy

**Thesis:** for an expensive simulator like LINX, *where* you spend your
simulation budget matters as much as the model. Active learning places new runs
where the calibrated emulator is least reliable, achieving target accuracy with
far fewer simulations than space-filling alone. This is the headline contribution.

## Loop
1. Start from a small Sobol' design (`init_size`).
2. Train an uncertainty-aware emulator (deep ensemble by default).
3. Score a large candidate pool with an acquisition function.
4. Select a batch (top-$k$, optional diversity penalty).
5. Query LINX at those points; add to the training set.
6. Retrain; evaluate on a fixed held-out test set; repeat for `rounds`.

See `run_active_learning` / `compare_acquisitions` in `bbnjax.active_learning`.

## Acquisition functions
| Name | Idea | Needs | Best for |
|---|---|---|---|
| `max_variance` | sample highest total predictive variance | any UQ model | simple, strong baseline |
| `bald` | maximize epistemic mutual information | ensemble/GP | avoids chasing noise |
| `integrated_variance` (ALC) | minimize *global* predictive variance | GP | smooth targets, principled |
| `random` | control baseline | — | quantifying the AL gain |

Formulas in [`math_derivations.md`](math_derivations.md) §7.

## Evaluation of the AL gain
The key plot (`viz.plots.plot_active_learning`) is **test error vs. number of
simulations**, comparing each acquisition against `random`. Report:
- simulations to reach a fixed accuracy (efficiency factor vs random),
- final accuracy at fixed budget,
- calibration (NLL / calibration error) across rounds — AL should not degrade UQ.

## Practical notes
- **Refit scalers each round** (the input distribution shifts as data is added);
  the loop rebuilds the `DataBundle` and trains fresh for reproducibility.
- **Diversity**: pure top-$k$ can cluster; enable the distance penalty in
  `select_batch` for batch acquisition.
- **Pool**: a large LHS pool approximates the prior volume; alternatively bias the
  pool toward the posterior region for *goal-oriented* AL (future work).
- **Cost model**: with LINX at ~0.03 s/run, thousands of queries are feasible; the
  benchmark suite quantifies emulator speedup at inference time.
