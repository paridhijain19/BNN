# Training strategy

## Data generation
- **Design:** Sobol' (default) for low-discrepancy coverage; LHS as a JAX-native
  alternative. Sizes are powers of two for Sobol' balance.
- **Ranges:** physically motivated, generous bounds (see `configs/default.yaml`)
  so the emulator covers beyond-current-constraint regions for forecasting.
- **Splits:** independent Sobol'/LHS designs for train/val/test (no leakage from
  shared low-discrepancy structure).

## Preprocessing
- Inputs standardized to zero mean / unit variance.
- Targets: `log10` for D/H, ³He/H, ⁷Li/H (orders of magnitude), then standardized.
  Scalers are fit on **train only** and stored with the model.

## Optimization
- **Optimizer:** AdamW with global-norm gradient clipping.
- **Schedule:** linear warmup (5%) → cosine decay to 1% of peak LR.
- **EMA:** exponential moving average of weights; the trainer keeps whichever of
  {raw, EMA} has the better validation loss each epoch (robust for short runs).
- **Early stopping:** patience on validation loss; best checkpoint restored.
- **Precision:** float64 throughout (BBN spans many orders of magnitude;
  Fisher matrices can be ill-conditioned).

## Model-specific notes
- **Residual emulator:** the linear baseline is fit closed-form and *frozen*
  (excluded from updates and weight decay) so the NN learns only the residual.
- **Deep ensemble:** members trained independently with (β-)NLL; β-NLL
  (Seitzer et al. 2022, β=0.5) stabilizes early heteroscedastic training.
- **NN+GP:** train the NN mean first, then fit GP hyperparameters by maximizing
  the marginal likelihood (Adam on log-hyperparameters); training data are held
  fixed (not optimized).

## Recommended recipe for a paper-quality emulator
1. Generate an initial Sobol' set (≈2–8k LINX runs).
2. Train the **residual** emulator (best point accuracy / sample efficiency).
3. Fit **NN+GP** (or ensemble) for calibrated uncertainty.
4. Run **active learning** to top up where the emulator is least reliable.
5. **Calibrate** on a held-out set; verify coverage.
6. Validate sensitivities (autodiff vs finite-difference) and benchmark vs LINX.

## Hyperparameters (sensible defaults)
| Knob | Default | Notes |
|---|---|---|
| width / depth | 128 / 4 | scale with data size |
| activation | GELU | smooth → better gradients/Fisher |
| LR | 1e-3 | AdamW |
| weight decay | 1e-5 | excludes frozen baseline |
| batch size | 256 | |
| epochs / patience | 400 / 60 | |
| GP kernel | Matérn-5/2 (ARD) | RBF available |
| ensemble members | 5 | |
