r"""Exact Gaussian-Process regression in JAX (ARD RBF / Matern-5/2 kernels).

Used as the *residual corrector* on top of a trained NN (see :mod:`nn_gp`) and,
on its own, as a fully probabilistic emulator for small designs. Provides the
marginal likelihood (for hyperparameter learning by gradient ascent) and the
posterior predictive mean/variance (for calibrated uncertainty and
active-learning acquisitions).

For :math:`n` training points with kernel matrix :math:`K`, noise :math:`\sigma_n^2`:

.. math::
    \log p(y\mid X) = -\tfrac12 y^\top \alpha
        - \sum_i \log L_{ii} - \tfrac n2\log 2\pi,\quad
    L=\mathrm{chol}(K+\sigma_n^2 I),\ \alpha=L^{-\top}L^{-1}y.

Predictive: :math:`\mu_* = k_*^\top\alpha`,
:math:`\sigma_*^2 = k_{**} - \lVert L^{-1}k_*\rVert^2`.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.linalg import cho_factor, cho_solve

_JITTER = 1e-8


def _scaled_dist_sq(x1: Array, x2: Array, lengthscale: Array) -> Array:
    """Pairwise squared distances under ARD lengthscales. ``x*`` are ``(n,d)``."""
    x1s = x1 / lengthscale
    x2s = x2 / lengthscale
    x1sq = jnp.sum(x1s**2, axis=1, keepdims=True)
    x2sq = jnp.sum(x2s**2, axis=1, keepdims=True)
    d2 = x1sq - 2.0 * x1s @ x2s.T + x2sq.T
    return jnp.clip(d2, 0.0, None)


def rbf_kernel(x1: Array, x2: Array, lengthscale: Array, signal_var: Array) -> Array:
    return signal_var * jnp.exp(-0.5 * _scaled_dist_sq(x1, x2, lengthscale))


def matern52_kernel(x1: Array, x2: Array, lengthscale: Array, signal_var: Array) -> Array:
    d2 = _scaled_dist_sq(x1, x2, lengthscale)
    r = jnp.sqrt(d2 + 1e-12)
    sqrt5 = jnp.sqrt(5.0)
    return signal_var * (1.0 + sqrt5 * r + 5.0 / 3.0 * d2) * jnp.exp(-sqrt5 * r)


_KERNELS = {"rbf": rbf_kernel, "matern52": matern52_kernel}


class GPModel(eqx.Module):
    """Single-output exact GP with a constant mean and ARD kernel."""

    X: Array
    y: Array
    log_lengthscale: Array
    log_signal: Array
    log_noise: Array
    mean: Array
    kernel: str = eqx.field(static=True)

    @classmethod
    def init(
        cls,
        X: Array,
        y: Array,
        *,
        kernel: str = "matern52",
        lengthscale: float = 1.0,
        signal_var: float = 1.0,
        noise: float = 1e-3,
    ) -> "GPModel":
        X = jnp.asarray(X)
        y = jnp.asarray(y)
        d = X.shape[1]
        return cls(
            X=X,
            y=y,
            log_lengthscale=jnp.log(jnp.full((d,), lengthscale)),
            log_signal=jnp.log(jnp.asarray(signal_var)),
            log_noise=jnp.log(jnp.asarray(noise)),
            mean=jnp.mean(y),
            kernel=kernel,
        )

    # -- internals --------------------------------------------------------------------

    def _params(self) -> tuple[Array, Array, Array]:
        return (
            jnp.exp(self.log_lengthscale),
            jnp.exp(self.log_signal),
            jnp.exp(self.log_noise),
        )

    def _kfun(self, x1: Array, x2: Array, ls: Array, sv: Array) -> Array:
        return _KERNELS[self.kernel](x1, x2, ls, sv)

    def _cho(self):
        ls, sv, noise = self._params()
        n = self.X.shape[0]
        K = self._kfun(self.X, self.X, ls, sv) + (noise + _JITTER) * jnp.eye(n)
        return cho_factor(K, lower=True), ls, sv

    # -- API --------------------------------------------------------------------------

    def log_marginal_likelihood(self) -> Array:
        (L, lower), ls, sv = self._cho()
        yc = self.y - self.mean
        alpha = cho_solve((L, lower), yc)
        n = self.X.shape[0]
        return (
            -0.5 * jnp.dot(yc, alpha)
            - jnp.sum(jnp.log(jnp.diag(L)))
            - 0.5 * n * jnp.log(2.0 * jnp.pi)
        )

    def predict(self, Xstar: Array) -> tuple[Array, Array]:
        """Posterior predictive ``(mean, variance)`` at ``Xstar`` (n*, ...)."""
        Xstar = jnp.atleast_2d(Xstar)
        (L, lower), ls, sv = self._cho()
        yc = self.y - self.mean
        alpha = cho_solve((L, lower), yc)
        Ks = self._kfun(self.X, Xstar, ls, sv)  # (n, n*)
        mean = self.mean + Ks.T @ alpha
        v = jax.scipy.linalg.solve_triangular(L, Ks, lower=True)
        kss = sv  # diagonal of k(x*, x*) for stationary kernels
        var = kss - jnp.sum(v**2, axis=0)
        var = jnp.clip(var, 1e-12, None)
        return mean, var


class MultiOutputGP(eqx.Module):
    """Independent GPs, one per output target (a 'diagonal' multi-output GP)."""

    gps: tuple

    @classmethod
    def init(cls, X: Array, Y: Array, **kwargs) -> "MultiOutputGP":
        Y = jnp.atleast_2d(jnp.asarray(Y))
        gps = tuple(GPModel.init(X, Y[:, j], **kwargs) for j in range(Y.shape[1]))
        return cls(gps=gps)

    def log_marginal_likelihood(self) -> Array:
        return sum(g.log_marginal_likelihood() for g in self.gps)

    def predict(self, Xstar: Array) -> tuple[Array, Array]:
        means, vars = [], []
        for g in self.gps:
            m, v = g.predict(Xstar)
            means.append(m)
            vars.append(v)
        return jnp.stack(means, axis=-1), jnp.stack(vars, axis=-1)


def _hyperparam_filter_spec(model):
    """Boolean filter PyTree marking only kernel hyperparameters as trainable.

    Crucially, the training data ``X``/``y`` are marked non-trainable so they are
    treated as fixed conditioning data, not optimized.
    """
    spec = jax.tree_util.tree_map(lambda _: False, model)
    if isinstance(model, GPModel):
        return eqx.tree_at(
            lambda m: [m.log_lengthscale, m.log_signal, m.log_noise, m.mean],
            spec,
            replace=[True, True, True, True],
        )
    if isinstance(model, MultiOutputGP):
        def where(m):
            leaves = []
            for g in m.gps:
                leaves += [g.log_lengthscale, g.log_signal, g.log_noise, g.mean]
            return leaves

        return eqx.tree_at(where, spec, replace=[True] * (4 * len(model.gps)))
    raise TypeError(f"Unsupported GP type for fitting: {type(model)}")


def fit_gp(gp, steps: int = 300, lr: float = 1e-2):
    """Maximize the (summed) log marginal likelihood w.r.t. kernel hyperparameters."""
    import optax

    filter_spec = _hyperparam_filter_spec(gp)
    diff, static = eqx.partition(gp, filter_spec)
    opt = optax.adam(lr)
    opt_state = opt.init(diff)

    @eqx.filter_jit
    def step(diff, static, opt_state):
        def loss_fn(d):
            model = eqx.combine(d, static)
            return -model.log_marginal_likelihood()

        loss, grads = eqx.filter_value_and_grad(loss_fn)(diff)
        updates, opt_state = opt.update(grads, opt_state, diff)
        diff = eqx.apply_updates(diff, updates)
        return diff, opt_state, loss

    losses = []
    for _ in range(steps):
        diff, opt_state, loss = step(diff, static, opt_state)
        losses.append(float(loss))
    return eqx.combine(diff, static), losses
