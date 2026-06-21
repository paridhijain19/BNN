"""Small PRNG helpers around ``jax.random`` for ergonomic, reproducible code."""

from __future__ import annotations

from typing import Iterator

import jax
from jax import Array


def split_key(key: Array, num: int = 2) -> tuple[Array, ...]:
    """Split a key into ``num`` subkeys (tuple of length ``num``)."""
    return tuple(jax.random.split(key, num))


class PRNGSequence:
    """A stateful, *explicitly seeded* stream of PRNG keys.

    JAX is functional, but threading keys through long training scripts is
    error-prone. This thin wrapper yields fresh keys on demand while keeping the
    whole stream deterministic given the seed. Use it only at the *orchestration*
    layer (training loops, data generation), never inside ``jit``-ed functions.

    Example
    -------
    >>> rng = PRNGSequence(0)
    >>> k1 = next(rng)
    >>> k2 = next(rng)
    """

    def __init__(self, seed: int | Array) -> None:
        if isinstance(seed, int):
            self._key = jax.random.PRNGKey(seed)
        else:
            self._key = seed

    def __next__(self) -> Array:
        self._key, subkey = jax.random.split(self._key)
        return subkey

    def __iter__(self) -> Iterator[Array]:
        return self

    def take(self, n: int) -> list[Array]:
        return [next(self) for _ in range(n)]
