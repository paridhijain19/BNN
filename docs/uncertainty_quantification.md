# Uncertainty quantification strategy

Reliable error bars are what make an emulator usable in inference — and what
drives active learning. BBN-JAX treats UQ as a first-class output.

## Sources of uncertainty
1. **Epistemic (model)** — what the emulator doesn't know due to limited training
   data. Captured by GP posterior variance and ensemble disagreement; it *grows
   away from training points* and is the signal active learning exploits.
2. **Aleatoric (irreducible)** — for a deterministic simulator this is ≈ the
   solver/interpolation noise; modeled by heteroscedastic ensemble heads / GP noise.
3. **Surrogate→science propagation** — emulator variance is optionally added to the
   data covariance in Fisher/likelihood so forecasts honestly include emulator error.

## Estimators
| Method | Epistemic | Aleatoric | Cost | When |
|---|---|---|---|---|
| NN+GP | GP posterior var | GP noise | GP $O(n^3)$ | small/medium designs, smooth |
| Deep ensemble | member spread | NLL heads | $M\times$ NN | large designs, GPU |

## Calibration
Raw variances are rarely calibrated. We fit a per-target variance scale
$s_a^2 = \frac1N\sum_i (y_{ia}-\mu_{ia})^2/\sigma_{ia}^2$ (closed-form NLL minimizer)
on a held-out set, giving $\tilde\sigma_a^2 = s_a^2\sigma_a^2$. $s_a\approx 1$ means
already calibrated; $s_a>1$ means over-confident.

## Diagnostics (in `uq.metrics`)
- **Reliability diagram** — observed vs expected central coverage; ideal = diagonal.
- **Calibration error** — mean $|{\rm observed}-{\rm expected}|$ over levels.
- **PICP@68/95** — interval coverage probabilities.
- **Standardized residuals** — $(y-\mu)/\sigma$ should be $\mathcal N(0,1)$.
- **Sharpness** — mean predictive std (smaller is better *once calibrated*).
- **NLL** — proper scoring rule trading off accuracy and calibration.

## Recommended protocol
1. Fit the UQ model; predict $(\mu,\sigma^2)$ on val + test.
2. Calibrate on val; **report test** reliability before/after.
3. Quote calibrated PICP, NLL, calibration error, and sharpness.
4. Propagate calibrated $\sigma_{\rm emu}^2$ into Fisher/likelihood; show it is
   subdominant to measurement errors (the bar for "good enough").
