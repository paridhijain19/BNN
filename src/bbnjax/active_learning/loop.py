r"""Active-learning loop: the central contribution of BBN-JAX.

Starting from a small space-filling design, we iteratively (i) train an
uncertainty-aware emulator, (ii) score a large candidate pool with an
acquisition function, (iii) query the (expensive) simulator at the most
informative points, and (iv) augment the training set. The whole loop is
benchmarked against random acquisition so the *simulation-efficiency* gain is
quantified — the headline plot of the paper.

The emulator used here is a :class:`DeepEnsemble` (fast epistemic variance);
``integrated_variance`` additionally fits a GP residual model to use the
closed-form ALC criterion.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from bbnjax.active_learning.acquisition import (
    gp_alc_scores,
    random_scores,
    score_bald,
    score_max_variance,
    select_batch,
)
from bbnjax.data.dataset import DataBundle
from bbnjax.models.ensemble import DeepEnsemble
from bbnjax.models.gp import MultiOutputGP, fit_gp
from bbnjax.models.wrapper import make_physical
from bbnjax.parameters import ParameterSpace
from bbnjax.sampling import make_design
from bbnjax.simulator import BBNSimulator, get_simulator
from bbnjax.training.trainer import train_ensemble
from bbnjax.uq import metrics
from bbnjax.uq.calibration import VarianceCalibrator
from bbnjax.utils.random import PRNGSequence


@dataclass
class ALHistory:
    acquisition: str
    n_train: list[int] = field(default_factory=list)
    test_rmse_mean: list[float] = field(default_factory=list)
    test_rmse_per_target: list[list[float]] = field(default_factory=list)
    test_nll: list[float] = field(default_factory=list)
    calibration_error: list[float] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "acquisition": self.acquisition,
            "n_train": self.n_train,
            "test_rmse_mean": self.test_rmse_mean,
            "test_rmse_per_target": self.test_rmse_per_target,
            "test_nll": self.test_nll,
            "calibration_error": self.calibration_error,
        }


def _build_bundle(space, C, Y, targets, log_targets, val=None) -> DataBundle:
    splits = {"train": {"C": np.asarray(C), "Y": np.asarray(Y)}}
    if val is not None:
        splits["val"] = {"C": np.asarray(val[0]), "Y": np.asarray(val[1])}
    return DataBundle.build(space, splits, target_names=targets, log_targets=log_targets)


def run_active_learning(cfg, sim: BBNSimulator | None = None, space: ParameterSpace | None = None):
    """Run one active-learning experiment defined by ``cfg`` and return its history."""
    space = space or ParameterSpace.from_config(cfg["parameters"])
    sim = sim or get_simulator(cfg["simulator"]["backend"])
    al = cfg["active_learning"]
    mcfg = cfg.get("model", {})
    tcfg = cfg.get("train", {})
    targets = tuple(cfg["targets"])
    log_targets = tuple(cfg.get("log_targets", ("DH", "He3H", "Li7H")))

    acquisition = al.get("acquisition", "max_variance")
    rounds = int(al.get("rounds", 6))
    batch_per_round = int(al.get("batch_per_round", 64))
    pool_size = int(al.get("pool_size", 4096))
    init_size = int(al.get("init_size", 256))
    retrain_epochs = int(al.get("retrain_epochs", 200))
    n_members = int(mcfg.get("n_members", 5))

    rng = PRNGSequence(int(cfg.get("seed", 0)))

    # Fixed evaluation + reference + validation designs (shared across rounds).
    test_C = space.from_unit(make_design("sobol", 512, space.dim, seed=10_001))
    test_Y = sim.predict_batch(test_C, space)
    val_C = space.from_unit(make_design("lhs", 128, space.dim, key=next(rng)))
    val_Y = sim.predict_batch(val_C, space)
    ref_C = space.from_unit(make_design("lhs", 256, space.dim, key=next(rng)))

    # Candidate pool and initial training set.
    pool_C = np.asarray(space.from_unit(make_design("lhs", pool_size, space.dim, key=next(rng))))
    available = np.ones(pool_size, dtype=bool)
    init_C = space.from_unit(make_design("sobol", init_size, space.dim, seed=777))
    train_C = np.asarray(init_C)
    train_Y = np.asarray(sim.predict_batch(init_C, space))

    history = ALHistory(acquisition=acquisition)

    for r in range(rounds + 1):
        bundle = _build_bundle(space, train_C, train_Y, targets, log_targets, val=(val_C, val_Y))
        X, T = bundle.xy("train")
        Xval, Tval = bundle.xy("val")

        ensemble = DeepEnsemble.init(
            space.dim,
            len(targets),
            n_members,
            width=int(mcfg.get("width", 128)),
            depth=int(mcfg.get("depth", 4)),
            activation=mcfg.get("activation", "gelu"),
            heteroscedastic=True,
            key=next(rng),
        )
        ensemble, _ = train_ensemble(
            ensemble,
            X,
            T,
            X_val=Xval,
            T_val=Tval,
            key=next(rng),
            epochs=retrain_epochs,
            batch_size=int(tcfg.get("batch_size", 256)),
            lr=float(tcfg.get("lr", 1e-3)),
            weight_decay=float(tcfg.get("weight_decay", 1e-5)),
            ema_decay=tcfg.get("ema_decay", 0.999),
        )

        # --- evaluate on the fixed test set (transformed space) ---
        Xtest = bundle.input_scaler.forward(jnp.asarray(test_C))
        Ttest = bundle.target_transform.forward(jnp.asarray(test_Y))
        mean_t, var_t = ensemble.predict_batch(Xtest)
        # Calibrate variance on validation, then evaluate NLL/coverage.
        mval, vval = ensemble.predict_batch(Xval)
        calib = VarianceCalibrator.fit(mval, vval, Tval)
        var_cal = calib.apply(var_t)

        rmse_pt = metrics.rmse(mean_t, Ttest)
        history.n_train.append(int(train_C.shape[0]))
        history.test_rmse_per_target.append([float(v) for v in rmse_pt])
        history.test_rmse_mean.append(float(jnp.mean(rmse_pt)))
        history.test_nll.append(float(metrics.gaussian_nll(mean_t, var_cal, Ttest, reduce=True)))
        history.calibration_error.append(metrics.calibration_error(mean_t, var_cal, Ttest))

        if r == rounds:
            break

        # --- score the available pool and select a query batch ---
        avail_idx = np.where(available)[0]
        pool_avail = jnp.asarray(pool_C[avail_idx])
        Xpool = bundle.input_scaler.forward(pool_avail)
        pmean, pvar = ensemble.predict_batch(Xpool)

        if acquisition == "random":
            scores = random_scores(next(rng), pool_avail.shape[0])
        elif acquisition == "max_variance":
            scores = score_max_variance(pvar)
        elif acquisition == "bald":
            # aleatoric = total minus epistemic; recompute per-member spread.
            member_means = jnp.stack([jax.vmap(m)(Xpool) for m in ensemble.members])
            epistemic = jnp.var(member_means, axis=0)
            aleatoric = jnp.clip(pvar - epistemic, 1e-12, None)
            scores = score_bald(pvar, aleatoric)
        elif acquisition == "integrated_variance":
            nn = ensemble.members[0]
            resid = T - jax.vmap(nn)(X)
            gp = MultiOutputGP.init(X, resid, kernel="matern52", noise=1e-3)
            gp, _ = fit_gp(gp, steps=150, lr=1e-2)
            Xref = bundle.input_scaler.forward(jnp.asarray(ref_C))
            scores = gp_alc_scores(gp, Xpool, Xref)
        else:
            raise ValueError(f"Unknown acquisition {acquisition!r}")

        sel_local = np.asarray(
            select_batch(scores, batch_per_round, pool_points=Xpool, diversity=0.0)
        )
        sel_global = avail_idx[sel_local]
        available[sel_global] = False

        new_C = pool_C[sel_global]
        new_Y = np.asarray(sim.predict_batch(jnp.asarray(new_C), space))
        train_C = np.concatenate([train_C, new_C], axis=0)
        train_Y = np.concatenate([train_Y, new_Y], axis=0)

    return history, bundle, ensemble


def compare_acquisitions(cfg, acquisitions=("max_variance", "random"), sim=None, space=None):
    """Run several acquisitions under identical settings; return ``{name: history}``."""
    space = space or ParameterSpace.from_config(cfg["parameters"])
    sim = sim or get_simulator(cfg["simulator"]["backend"])
    out = {}
    for acq in acquisitions:
        cfg_acq = dict(cfg)
        al = dict(cfg["active_learning"])
        al["acquisition"] = acq
        cfg_acq["active_learning"] = al
        history, _, _ = run_active_learning(cfg_acq, sim=sim, space=space)
        out[acq] = history
    return out
