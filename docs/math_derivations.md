# Mathematical derivations

Notation: cosmological parameters $\mathbf c \in \mathbb R^D$ (e.g. $\Omega_b h^2,
N_{\rm eff}, \tau_n, \xi_e$), abundances $\mathbf Y \in \mathbb R^T$ with
$T=4$ targets $(Y_p, \mathrm{D/H}, {}^3\mathrm{He/H}, {}^7\mathrm{Li/H})$. The
ground-truth simulator is $\mathcal S:\mathbf c \mapsto \mathbf Y$ (LINX).

## 1. Target and input transforms

Wide-dynamic-range abundances are emulated in $\log_{10}$ space, then standardized:

$$ t_a(\mathbf c) = \frac{g_a(Y_a(\mathbf c)) - \mu_a}{s_a},\qquad
g_a = \begin{cases}\log_{10} & a \in \{\mathrm{D/H}, {}^3\mathrm{He/H}, {}^7\mathrm{Li/H}\}\\ \mathrm{id} & a = Y_p\end{cases} $$

with $\mu_a, s_a$ the train-set mean/std of $g_a(Y_a)$. Inputs are standardized
$x_i = (c_i - \bar c_i)/\sigma_{c_i}$. The emulator learns $f_\theta: \mathbf x \mapsto \mathbf t$,
and physical predictions are recovered by inverting both transforms:
$\hat Y_a = g_a^{-1}(s_a f_{\theta,a}(\mathbf x) + \mu_a)$.

## 2. Emulator families

### 2.1 MLP
A standard MLP $f_\theta$ trained by mean-squared error
$\mathcal L_{\rm MSE} = \frac1{NT}\sum_{n,a}(f_{\theta,a}(\mathbf x_n) - t_{na})^2$.

### 2.2 Residual (physics + NN)
Let $B(\mathbf x)=W\mathbf x + b$ be a ridge least-squares baseline fit in closed form,
$ \min_{W,b}\ \lVert X[W^\top;b^\top] - T\rVert_F^2 + \lambda\lVert\cdot\rVert^2 $.
The emulator is $\hat{\mathbf t}(\mathbf x) = \mathrm{sg}[B(\mathbf x)] + s\,\mathrm{NN}_\theta(\mathbf x)$,
where $\mathrm{sg}$ is stop-gradient (the baseline is frozen). The NN only models
the residual curvature, improving sample efficiency.

### 2.3 Gaussian process
For one target with training inputs $X$, targets $y$, kernel $k$ (ARD Matérn-5/2
by default) and noise $\sigma_n^2$, define $K_{ij}=k(\mathbf x_i,\mathbf x_j)$,
$L=\mathrm{chol}(K+\sigma_n^2 I)$, $\alpha=L^{-\top}L^{-1}(y-m)$. The **log marginal
likelihood** (maximized for hyperparameters) is

$$ \log p(y\mid X) = -\tfrac12 (y-m)^\top\alpha - \sum_i\log L_{ii} - \tfrac N2\log 2\pi. $$

The **posterior predictive** at $\mathbf x_*$ with $k_*=k(X,\mathbf x_*)$ is

$$ \mu_* = m + k_*^\top\alpha,\qquad
\sigma_*^2 = k(\mathbf x_*,\mathbf x_*) - \lVert L^{-1}k_*\rVert_2^2. $$

### 2.4 NN + GP residual
Train the NN mean $m_{\rm NN}$, form residuals $r_n = t_n - m_{\rm NN}(\mathbf x_n)$,
fit a (per-target) GP to $r$. Then

$$ \hat t(\mathbf x) = m_{\rm NN}(\mathbf x) + \mu_{\rm GP}(\mathbf x),\qquad
\mathrm{Var}[\hat t(\mathbf x)] = \sigma_{\rm GP}^2(\mathbf x). $$

### 2.5 Deep ensemble
$M$ heteroscedastic nets, each emitting $(\mu_m, \sigma_m^2)$, trained by Gaussian NLL.
The mixture mean/variance are

$$ \mu = \tfrac1M\sum_m\mu_m,\quad
\sigma^2 = \underbrace{\tfrac1M\sum_m\sigma_m^2}_{\text{aleatoric}}
        + \underbrace{\tfrac1M\sum_m(\mu_m-\mu)^2}_{\text{epistemic}}. $$

## 3. Uncertainty calibration

Given a calibration set, rescale variances per target, $\tilde\sigma_a^2 = s_a^2\sigma_a^2$.
Minimizing the Gaussian NLL over $s_a$,
$\frac{\partial}{\partial s_a^2}\sum_i\big[\log(s_a^2\sigma_{ia}^2) + \frac{(y_{ia}-\mu_{ia})^2}{s_a^2\sigma_{ia}^2}\big]=0$,
gives the closed form

$$ s_a^2 = \frac1N\sum_i \frac{(y_{ia}-\mu_{ia})^2}{\sigma_{ia}^2}, $$

i.e. the mean squared standardized residual ($s_a=1$ ⇔ already calibrated).

## 4. Autodiff sensitivities

Because $\hat{\mathbf Y}(\mathbf c)$ is differentiable, the Jacobian is exact:
$J_{ai} = \partial \hat Y_a/\partial c_i$ via `jax.jacfwd` (forward mode, small $D$).
Dimensionless log-sensitivities (standard in BBN) are
$\partial\ln Y_a/\partial\ln c_i = J_{ai}\, c_i / Y_a$.

## 5. Fisher forecast

For a Gaussian likelihood with parameter-independent covariance $\Sigma$
(measurement errors $\oplus$ optional emulator variance), the Fisher information is

$$ F_{ij} = (J^\top\Sigma^{-1}J)_{ij}
= \sum_{ab}\frac{\partial Y_a}{\partial c_i}(\Sigma^{-1})_{ab}\frac{\partial Y_b}{\partial c_j}. $$

Gaussian priors add $1/\sigma_{\pi,i}^2$ to the diagonal. Parameter covariance is
$C=F^{-1}$; marginalized errors $\sigma_i=\sqrt{C_{ii}}$, conditional errors
$1/\sqrt{F_{ii}}$, with $\sigma_i^{\rm marg}\ge\sigma_i^{\rm cond}$ always.

## 6. Likelihood

For observed abundances $\hat Y_a$ with variances $\sigma_a^2$ (plus emulator
$\sigma_{{\rm emu},a}^2$),

$$ \ln\mathcal L(\mathbf c) = -\tfrac12\sum_{a\in\mathcal O}
\left[\frac{(\hat Y_a - Y_a(\mathbf c))^2}{\sigma_a^2+\sigma_{{\rm emu},a}^2}
+ \ln\!\big(2\pi(\sigma_a^2+\sigma_{{\rm emu},a}^2)\big)\right]. $$

$\nabla_{\mathbf c}\ln\mathcal L$ is exact via autodiff (HMC/NUTS-ready).

## 7. Active-learning acquisitions

Let predictive variance over a pool be $\sigma_a^2(\mathbf c)$ (epistemic $+$ aleatoric).

- **Uncertainty sampling:** $\;a(\mathbf c)=\sum_a w_a\sigma_a^2(\mathbf c)$.
- **BALD (mutual information), Gaussian approx:**
  $\;\mathcal I(\mathbf c)\approx \tfrac12\sum_a\log\!\frac{\sigma_a^2(\mathbf c)}{\sigma_{a,{\rm alea}}^2(\mathbf c)}$ —
  rewards *epistemic* uncertainty only.
- **ALC / integrated variance (GP):** the variance reduction at reference point
  $\mathbf x_r$ from adding candidate $\mathbf x_c$ is
  $\Delta\sigma^2(\mathbf x_r;\mathbf x_c)=\frac{k(\mathbf x_r,\mathbf x_c)^2}{k(\mathbf x_c,\mathbf x_c)+\sigma_n^2}$;
  the acquisition integrates this over a reference set,
  $a(\mathbf x_c)=\sum_r\Delta\sigma^2(\mathbf x_r;\mathbf x_c)$.

Batches are chosen greedily (top-$k$) with an optional distance penalty to
preserve diversity.
