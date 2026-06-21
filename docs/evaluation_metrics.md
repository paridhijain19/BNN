# Evaluation metrics

All metrics live in `bbnjax.uq.metrics` and accept arrays `(N, T)` (samples ×
targets), returning per-target vectors (or a scalar mean with `reduce=True`).

## Point accuracy
| Metric | Definition | Notes |
|---|---|---|
| RMSE | $\sqrt{\frac1N\sum(\hat y-y)^2}$ | in standardized or physical units |
| MAE | $\frac1N\sum|\hat y-y|$ | robust to outliers |
| Max abs error | $\max|\hat y-y|$ | worst case (matters for tails) |
| $R^2$ | $1-\mathrm{SS_{res}}/\mathrm{SS_{tot}}$ | variance explained |
| Median / max **relative** error | $|\hat Y-Y|/|Y|$ | physical-space, per target (benchmark) |

**Targets for a science-ready emulator:** emulator error well below measurement
precision — e.g. relative error ≲ 0.1–0.3% on $Y_p$ and D/H (LINX↔PRyMordial
agreement is already <0.05–0.15%, so the emulator must not dominate the budget).

## Probabilistic / calibration
| Metric | Meaning |
|---|---|
| Gaussian NLL | proper scoring rule (accuracy + calibration) |
| PICP@68 / @95 | fraction inside central intervals (target = nominal) |
| Reliability curve | observed vs expected coverage (ideal = diagonal) |
| Calibration error | mean abs deviation from the diagonal |
| Sharpness | mean predictive std (lower better, *post*-calibration) |
| Standardized residuals | should be $\mathcal N(0,1)$ |

## Differentiable-inference checks
- **Autodiff vs finite-difference Jacobian**: max relative discrepancy should be
  ≲ $10^{-3}$ (smaller for the analytic mock). Validates gradient correctness.
- **Fisher sanity**: symmetric, positive semi-definite; marginal ≥ conditional
  errors; priors shrink errors.

## Benchmark (`benchmarks/benchmark_emulator.py`)
- **Throughput**: wall-time vs number of evaluations, LINX vs emulator → speedup.
- **Accuracy**: emulator vs simulator relative error on a fresh design.

## Reporting checklist for the paper
- [ ] Point accuracy table (per target, physical relative error)
- [ ] Calibration (reliability + PICP + NLL), before/after calibration
- [ ] Active-learning efficiency curve vs random (with seeds/error bars)
- [ ] Sensitivity figure (autodiff) + FD validation number
- [ ] Fisher forecast + emulator-error propagation (subdominant)
- [ ] Benchmark speedup vs LINX
- [ ] Seeds, versions, git SHA (from run manifests)
