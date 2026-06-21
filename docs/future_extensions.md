# Future extensions

Ordered roughly by impact/effort.

## Modeling
- **Multi-output / correlated GP.** Replace independent per-target GPs with a
  linear model of coregionalization or an intrinsic coregionalization model to
  exploit cross-abundance correlations (e.g. $Y_p$–D/H via $\Omega_b h^2$).
- **Nuclear-rate nuisance inputs.** Add LINX `nuclear_rates_q` (the $q_i$
  reaction-rate offsets) as emulator inputs, enabling marginalization over nuclear
  uncertainties — especially important for ⁷Li.
- **Neutrino degeneracy $\xi_e$.** Integrate a degeneracy-aware LINX background /
  weak-rate scenario so $\xi_e$ becomes a first-class, supported parameter.
- **Beyond-SBBN physics.** Dark radiation beyond constant $\Delta N_{\rm eff}$,
  time-varying constants, decaying particles — the modular simulator API supports
  swapping scenarios.
- **Normalizing-flow / heteroscedastic-flow heads** for non-Gaussian predictive
  distributions where abundances are skewed.

## Active learning
- **Goal-oriented (Bayesian) AL.** Bias the candidate pool toward the posterior
  (or Fisher-informative directions) instead of the full prior volume.
- **Batch acquisition via determinantal point processes** for principled diversity.
- **Cost-aware AL.** Weight acquisition by per-point simulation cost (key vs full
  network, integration tolerance).
- **Multi-fidelity.** Combine cheap "key"-network and expensive "full"-network
  LINX runs in a multi-fidelity GP / AL scheme.

## Inference
- **Full posteriors with blackjax** (NUTS using the exact emulator gradients) and
  simulation-based inference baselines.
- **Joint CMB+BBN.** Couple with a differentiable CMB emulator for joint
  $\Omega_b h^2$/$N_{\rm eff}$ constraints (a stated LINX use case).
- **Second-order Fisher / DALI** using exact Hessians (`jax.hessian`).

## Software / reproducibility
- **`eqx.tree_serialise_leaves` + architecture spec** for long-term, framework-
  independent checkpoints (in addition to pickle).
- **W&B / MLflow** tracker backends behind the `ExperimentTracker` interface.
- **GPU/TPU benchmarks** and `pmap`/`shard_map` data generation at scale.
- **Hugging Face / Zenodo** release of pretrained emulators with versioned cards.
- **CI matrix** (Linux/macOS/Windows, CPU/GPU) and docs site (mkdocs/Sphinx).

## Science studies enabled
- Updated BBN parameter constraints with calibrated emulator error budgets.
- Sensitivity atlas of $\partial\ln Y_a/\partial\ln c_i$ across the parameter space.
- Robustness of the lithium problem to nuclear-rate marginalization.
