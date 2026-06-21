# Publication roadmap

**Working title:** *Active-learning, uncertainty-aware, differentiable Big Bang
Nucleosynthesis emulation in JAX, built on LINX.*

**One-line claim:** a calibrated, differentiable BBN surrogate whose own
uncertainty selects training simulations, reaching target accuracy with markedly
fewer LINX runs than space-filling, and exposing exact Jacobians for Fisher and
gradient-based cosmological inference.

## Why it's novel (vs. prior emulators)
- Most BBN/cosmology emulators are trained on fixed grids; here the **acquisition
  is uncertainty-driven and benchmarked vs random** (quantified efficiency gain).
- **End-to-end differentiability** in physical units → exact sensitivities,
  Fisher, and HMC scores (not finite differences).
- **Calibrated UQ propagated into forecasts**, showing emulator error is
  subdominant — a reliability standard often skipped.
- Built natively on **LINX**, itself differentiable, enabling apples-to-apples
  gradient validation.

## Target venues
- Methods/ML-for-science: *JCAP*, *ApJ/ApJS*, *MNRAS*; ML venues: *NeurIPS/ICML
  ML4PS workshop*, *MLST*.
- Companion: open-source package (this repo) with Zenodo DOI.

## Experimental plan
1. **Ground truth.** Generate LINX datasets (key + full networks) across
   $\Omega_b h^2, N_{\rm eff}, \tau_n$ (and $\xi_e$ via a degeneracy-aware
   scenario; see limitations). Multiple seeds.
2. **Emulator comparison.** MLP vs residual vs NN+GP vs ensemble: accuracy,
   calibration, training cost. Establish best base model.
3. **Active learning.** Sweep acquisitions (`max_variance`, `bald`,
   `integrated_variance`) vs `random`; report simulations-to-accuracy and final
   accuracy at fixed budget, with seed-averaged error bars. **Headline result.**
4. **UQ validation.** Reliability/PICP/NLL before vs after calibration; coverage
   on out-of-distribution slices.
5. **Differentiable inference.** Autodiff sensitivities (vs FD), Fisher forecasts
   with/without emulator-error propagation; optionally a full HMC/NUTS posterior
   (blackjax) and a Cobaya BBN-only run using the wrapper.
6. **Benchmark.** Speedup vs LINX at inference; memory; GPU scaling.

## Figures (auto-produced by `scripts/08_make_figures.py`)
- Predicted-vs-true; residual histograms.
- Reliability diagram (raw vs calibrated).
- **Active-learning efficiency curve** (main figure).
- Sensitivity bar chart (autodiff).
- Fisher corner plot.
- LINX-vs-emulator throughput.

## Milestones
| Phase | Deliverable |
|---|---|
| M1 | LINX data generation + reproducible pipeline (this repo) |
| M2 | Best base emulator + calibration results |
| M3 | Active-learning efficiency study (headline) |
| M4 | Differentiable inference + Cobaya/HMC demo |
| M5 | Paper draft + package release (Zenodo DOI) |

## Limitations to state plainly
- The shipped **mock** is illustrative; all numbers must come from **LINX**.
- $\xi_e$ (neutrino degeneracy) needs a degeneracy-aware LINX scenario; stock SBBN
  LINX does not support it (the adapter raises by default).
- ⁷Li requires the **full** network for accuracy and carries large nuclear-rate
  uncertainty (the "lithium problem").
- Independent-output GP ignores cross-target correlations (see future extensions).
