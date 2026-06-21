"""Generic Equinox training loop with AdamW, EMA, and early stopping.

Handles parameter freezing automatically (e.g. the frozen linear baseline inside
:class:`ResidualEmulator` is excluded from updates *and* weight decay), supports
arbitrary per-batch loss functions, and tracks the best validation checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jax import Array

from bbnjax.models.ensemble import DeepEnsemble
from bbnjax.models.residual import ResidualEmulator
from bbnjax.training.losses import gaussian_nll_loss, mse_loss
from bbnjax.training.schedules import build_optimizer


@dataclass
class TrainResult:
    model: eqx.Module
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    best_val: float = float("inf")
    best_epoch: int = -1


def _trainable_filter(model: eqx.Module):
    """Boolean filter: all inexact arrays are trainable except frozen baselines."""
    spec = jax.tree_util.tree_map(eqx.is_inexact_array, model)
    if isinstance(model, ResidualEmulator):
        spec = eqx.tree_at(
            lambda m: m.baseline,
            spec,
            replace=jax.tree_util.tree_map(lambda _: False, model.baseline),
        )
    return spec


def train_model(
    model: eqx.Module,
    X: Array,
    T: Array,
    *,
    X_val: Array | None = None,
    T_val: Array | None = None,
    loss_fn=mse_loss,
    epochs: int = 400,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    grad_clip: float = 1.0,
    warmup_frac: float = 0.05,
    ema_decay: float | None = 0.999,
    patience: int | None = None,
    key: Array,
    verbose: bool = False,
) -> TrainResult:
    """Train ``model`` on ``(X, T)`` returning the best (EMA) checkpoint."""
    X = jnp.asarray(X)
    T = jnp.asarray(T)
    n = X.shape[0]
    steps_per_epoch = max(1, n // batch_size)
    total_steps = epochs * steps_per_epoch
    optim = build_optimizer(
        lr, total_steps, weight_decay=weight_decay, grad_clip=grad_clip, warmup_frac=warmup_frac
    )

    filter_spec = _trainable_filter(model)
    diff, static = eqx.partition(model, filter_spec)
    opt_state = optim.init(diff)
    use_ema = ema_decay is not None
    ema = diff

    @eqx.filter_jit
    def step(diff, opt_state, ema, xb, tb):
        def loss_only(d):
            return loss_fn(eqx.combine(d, static), xb, tb)

        loss, grads = eqx.filter_value_and_grad(loss_only)(diff)
        updates, opt_state = optim.update(grads, opt_state, diff)
        diff = eqx.apply_updates(diff, updates)
        if use_ema:
            ema = jax.tree_util.tree_map(
                lambda e, p: ema_decay * e + (1.0 - ema_decay) * p, ema, diff
            )
        return diff, opt_state, ema, loss

    @eqx.filter_jit
    def eval_loss(diff, xb, tb):
        return loss_fn(eqx.combine(diff, static), xb, tb)

    result = TrainResult(model=model)
    best_diff = ema if use_ema else diff
    no_improve = 0

    for epoch in range(epochs):
        key, sub = jax.random.split(key)
        perm = jax.random.permutation(sub, n)
        epoch_losses = []
        for s in range(steps_per_epoch):
            idx = perm[s * batch_size : (s + 1) * batch_size]
            diff, opt_state, ema, loss = step(diff, opt_state, ema, X[idx], T[idx])
            epoch_losses.append(float(loss))
        result.train_loss.append(float(jnp.mean(jnp.asarray(epoch_losses))))

        if X_val is not None and T_val is not None:
            # Evaluate both the raw and EMA weights and keep whichever is better;
            # this is robust when few steps make the EMA lag the raw weights.
            candidates = [(diff, float(eval_loss(diff, X_val, T_val)))]
            if use_ema:
                candidates.append((ema, float(eval_loss(ema, X_val, T_val))))
            cand_diff, vloss = min(candidates, key=lambda kv: kv[1])
            result.val_loss.append(vloss)
            if vloss < result.best_val - 1e-9:
                result.best_val = vloss
                result.best_epoch = epoch
                best_diff = cand_diff
                no_improve = 0
            else:
                no_improve += 1
            if verbose and epoch % max(1, epochs // 10) == 0:
                print(f"  epoch {epoch:4d}  train={result.train_loss[-1]:.4e}  val={vloss:.4e}")
            if patience is not None and no_improve >= patience:
                if verbose:
                    print(f"  early stop at epoch {epoch} (best={result.best_val:.4e})")
                break
        else:
            best_diff = ema if use_ema else diff

    result.model = eqx.combine(best_diff, static)
    return result


def train_ensemble(
    ensemble: DeepEnsemble,
    X: Array,
    T: Array,
    *,
    X_val: Array | None = None,
    T_val: Array | None = None,
    key: Array,
    beta_nll: float = 0.5,
    **kwargs,
) -> tuple[DeepEnsemble, list[TrainResult]]:
    """Train each ensemble member independently with the (beta-)NLL loss."""
    members = []
    results = []
    keys = jax.random.split(key, len(ensemble.members))
    loss_fn = lambda m, x, t: gaussian_nll_loss(m, x, t, beta=beta_nll)
    for member, k in zip(ensemble.members, keys):
        res = train_model(
            member, X, T, X_val=X_val, T_val=T_val, loss_fn=loss_fn, key=k, **kwargs
        )
        members.append(res.model)
        results.append(res)
    return DeepEnsemble(members=tuple(members)), results
