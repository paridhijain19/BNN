# BBN-JAX

**Differentiable, uncertainty-aware, active-learning machine-learning emulators for Big Bang Nucleosynthesis, built on [LINX](https://github.com/cmbant/LINX).**

> Core contribution: **Active-learning, uncertainty-aware, *differentiable* BBN emulation in JAX built on LINX.**
> Not just an NN+GP surrogate — an end-to-end differentiable pipeline whose calibrated uncertainties drive where new training simulations are placed, and whose Jacobians feed directly into Fisher forecasts and gradient-based cosmological inference.

---

## Why this project

Big Bang Nucleosynthesis (BBN) is one of the most precise probes of the early Universe. Predicting the
primordial abundances (\(Y_p\), D/H, \(^3\mathrm{He}/\mathrm{H}\), \(^7\mathrm{Li}/\mathrm{H}\)) from cosmological
parameters requires integrating a stiff thermonuclear reaction network. Even with the fast JAX-native
solver **LINX**, repeated evaluation inside MCMC / nested-sampling / Fisher pipelines remains a bottleneck,
and gradients through the full network can be expensive or noisy.

**BBN-JAX** learns a fast, smooth, *fully differentiable* surrogate \( f_\theta:\ \mathbf{c}\mapsto \mathbf{Y} \)
mapping cosmological parameters \(\mathbf{c}\) to abundances \(\mathbf{Y}\), with:

- **calibrated predictive uncertainties** (so you know when to trust it),
- **active learning** (so simulation budget is spent where the emulator is least reliable),
- **exact autodiff sensitivities** \( \partial \mathbf{Y}/\partial \mathbf{c}\) for Fisher forecasts and HMC/NUTS,
- a **Cobaya-compatible likelihood** for drop-in cosmological analyses.

## Parameters and observables

| Inputs \(\mathbf{c}\)                          | Symbol            | Notes                                            |
|------------------------------------------------|-------------------|--------------------------------------------------|
| Baryon density                                 | \(\Omega_b h^2\)  | equivalently the baryon-to-photon ratio \(\eta_b\) |
| Baryon-to-photon ratio                         | \(\eta_b\)        | \(\eta_{10}=10^{10}\eta_b \approx 273.9\,\Omega_b h^2\) |
| Effective number of neutrino species           | \(N_{\rm eff}\)   | standard value \(\approx 3.044\)                 |
| Neutron lifetime                               | \(\tau_n\)        | seconds; PDG \(\approx 878.4\)                    |
| Electron-neutrino degeneracy                   | \(\xi_e\)         | chemical potential \(\mu_{\nu_e}/T_\nu\)         |
| Dark radiation (optional)                      | \(\Delta N_{\rm dark}\) | extra relativistic species                |

| Outputs \(\mathbf{Y}\)         | Symbol                       | Emulated in                |
|--------------------------------|------------------------------|----------------------------|
| Helium-4 mass fraction         | \(Y_p\)                      | linear                     |
| Deuterium                      | \(\mathrm{D/H}\)             | \(\log_{10}\)              |
| Helium-3                       | \(^3\mathrm{He}/\mathrm{H}\) | \(\log_{10}\)              |
| Lithium-7                      | \(^7\mathrm{Li}/\mathrm{H}\) | \(\log_{10}\)              |

## Highlights

- **JAX-native end to end** — sampling, simulation adapter, models, training, UQ, sensitivities, Fisher.
- **Equinox models** — MLP, residual emulator (`physics + NN`), deep ensembles.
- **NN + Gaussian-Process residual** — exact JAX GP corrects the NN and supplies a principled error bar.
- **Calibrated UQ** — temperature/variance scaling with reliability diagrams and coverage tests.
- **Active learning** — variance / BALD / integrated-variance acquisition over the prior volume.
- **Differentiable inference** — `jax.jacfwd`/`jacrev` sensitivities, Fisher matrices, Cobaya likelihood.
- **Reproducible** — config-driven runs, seeded PRNG, experiment manifests, deterministic benchmarks.
- **Runs without LINX** — a physically-motivated analytic mock simulator ships for CI and demos.

## Repository layout

```
BNN/
├── src/bbnjax/
│   ├── parameters.py        # parameter space, priors, transforms, fiducials
│   ├── sampling.py          # Sobol & Latin Hypercube designs
│   ├── config.py            # YAML-backed dataclass configs
│   ├── simulator/           # base API, LINX adapter, analytic mock
│   ├── data/                # dataset generation, containers, scalers
│   ├── models/              # MLP, residual, GP, NN+GP, ensembles
│   ├── training/            # losses, schedules, trainer, state
│   ├── uq/                  # calibration + UQ metrics
│   ├── active_learning/     # acquisition functions + AL loop
│   ├── inference/           # sensitivities, Fisher, Cobaya likelihood
│   ├── tracking/            # experiment manifests / metric logs
│   ├── viz/                 # publication figures
│   └── utils/               # PRNG, IO helpers
├── scripts/                 # 00..08 end-to-end pipeline
├── configs/                 # YAML configs for every stage
├── tests/                   # pytest unit tests
├── benchmarks/              # LINX-vs-emulator timing & accuracy
├── docs/                    # architecture, derivations, strategies, roadmap
└── pyproject.toml
```

## Installation

```bash
# create / activate an environment (conda or venv), then:
pip install -e ".[dev]"          # core + dev tooling
# optional extras:
pip install -e ".[all]"          # tinygp, blackjax
pip install linx                 # ground-truth simulator (otherwise the mock is used)
```

GPU users: install the matching `jax[cuda12]` wheel for your CUDA version instead of the CPU default.

## Quickstart

```bash
# 1. Generate a (mock or LINX) training set via Latin-Hypercube design
python scripts/00_generate_data.py --config configs/data_gen.yaml

# 2. Train the baseline MLP emulator
python scripts/01_train_mlp.py --config configs/train_mlp.yaml

# 3. Train the residual (physics + NN) emulator
python scripts/02_train_residual.py --config configs/train_residual.yaml

# 4. Fit the NN + GP residual correction and calibrate uncertainties
python scripts/03_train_nn_gp.py
python scripts/04_calibrate_uq.py

# 5. Run an active-learning round
python scripts/05_active_learning.py --config configs/active_learning.yaml

# 6. Fisher forecast + benchmark + figures
python scripts/06_fisher_forecast.py
python scripts/07_benchmark.py
python scripts/08_make_figures.py
```

Minimal Python API:

```python
import jax
from bbnjax.parameters import ParameterSpace
from bbnjax.simulator import get_simulator

space = ParameterSpace.default()
sim = get_simulator("auto")              # LINX if installed, else analytic mock
c = space.fiducial_vector()
Y = sim.predict(c)                       # -> {"Yp", "DH", "He3H", "Li7H"}

# Differentiable sensitivities of the emulator
from bbnjax.inference.sensitivity import abundance_jacobian
J = abundance_jacobian(emulator, c)      # dY/dc, exact via autodiff
```

## Documentation

- [`docs/architecture.md`](docs/architecture.md) — system design + diagrams
- [`docs/math_derivations.md`](docs/math_derivations.md) — emulator, GP, calibration, Fisher math
- [`docs/training_strategy.md`](docs/training_strategy.md)
- [`docs/uncertainty_quantification.md`](docs/uncertainty_quantification.md)
- [`docs/active_learning.md`](docs/active_learning.md)
- [`docs/evaluation_metrics.md`](docs/evaluation_metrics.md)
- [`docs/publication_roadmap.md`](docs/publication_roadmap.md)
- [`docs/future_extensions.md`](docs/future_extensions.md)

## Scientific status & honesty note

The shipped **mock simulator** uses published-style *linearized fitting functions* so the whole pipeline is
runnable and testable in CI without LINX. **Quantitative scientific results in any paper must be produced with
the real LINX backend** (`get_simulator("linx")`); the mock exists only for software testing, demos, and as a
known-ground-truth target for validating the emulator/active-learning machinery.

## Citing

See [`CITATION.cff`](CITATION.cff). Please also cite LINX.

## License

MIT — see [`LICENSE`](LICENSE).
