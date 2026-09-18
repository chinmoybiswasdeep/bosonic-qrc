from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from .config import DVConfig
from .experiment import run_xor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/calibration/xor"))
    args = parser.parse_args()
    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    manifest = run_xor(DVConfig(**raw["reservoir"]), args.output)
    print(manifest["train_metrics"])


if __name__ == "__main__":
    main()
