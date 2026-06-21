"""Stage 00: generate a BBN emulator dataset from a space-filling design."""

from __future__ import annotations

import argparse

from _common import load_config
from bbnjax.data.generate import generate_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data_gen.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    generate_dataset(cfg)


if __name__ == "__main__":
    main()
