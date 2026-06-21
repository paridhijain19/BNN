"""Lightweight YAML config loading with ``defaults`` inheritance and dotted access.

A config file may declare ``defaults: <relative-path.yaml>`` to inherit and
deep-merge another file (the current file's keys win). This keeps stage configs
small while sharing the global parameter/target definitions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml


class Config(dict):
    """A ``dict`` with attribute access and recursive wrapping.

    >>> c = Config({"a": {"b": 1}})
    >>> c.a.b
    1
    >>> c.get("missing", 42)
    42
    """

    def __getattr__(self, item: str) -> Any:
        try:
            value = self[item]
        except KeyError as exc:  # pragma: no cover - error path
            raise AttributeError(item) from exc
        if isinstance(value, dict) and not isinstance(value, Config):
            value = Config(value)
            self[item] = value
        return value

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def _deep_merge(base: dict, override: Mapping) -> dict:
    out = dict(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, Mapping):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path) -> Config:
    """Load a YAML config, resolving a single-level ``defaults:`` chain.

    The ``defaults`` path is resolved relative to the file that declares it.
    """
    path = Path(path)
    with open(path, encoding="utf-8") as fh:
        raw: dict = yaml.safe_load(fh) or {}

    defaults_rel = raw.pop("defaults", None)
    if defaults_rel is not None:
        base = load_config(path.parent / defaults_rel)
        merged = _deep_merge(dict(base), raw)
    else:
        merged = raw
    return Config(merged)
