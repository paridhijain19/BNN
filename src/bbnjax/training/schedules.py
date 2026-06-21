"""Optimizer / learning-rate schedule construction (Optax)."""

from __future__ import annotations

import optax


def warmup_cosine(
    lr: float,
    total_steps: int,
    warmup_frac: float = 0.05,
    end_lr_frac: float = 0.01,
) -> optax.Schedule:
    """Linear warmup then cosine decay to ``end_lr_frac * lr``."""
    warmup_steps = max(1, int(warmup_frac * total_steps))
    decay_steps = max(1, total_steps - warmup_steps)
    return optax.join_schedules(
        schedules=[
            optax.linear_schedule(0.0, lr, warmup_steps),
            optax.cosine_decay_schedule(lr, decay_steps, alpha=end_lr_frac),
        ],
        boundaries=[warmup_steps],
    )


def build_optimizer(
    lr: float,
    total_steps: int,
    *,
    weight_decay: float = 1e-5,
    grad_clip: float = 1.0,
    warmup_frac: float = 0.05,
) -> optax.GradientTransformation:
    """AdamW with global-norm clipping and a warmup-cosine schedule."""
    schedule = warmup_cosine(lr, total_steps, warmup_frac=warmup_frac)
    return optax.chain(
        optax.clip_by_global_norm(grad_clip),
        optax.adamw(learning_rate=schedule, weight_decay=weight_decay),
    )
