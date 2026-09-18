"""Command line interface."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from .config import CVConfig
from .experiment import save_multiseed_run, save_run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/cv"))
    args = parser.parse_args()
    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    config = CVConfig(**raw.pop("reservoir"))
    if "seeds" in raw:
        manifest = save_multiseed_run(
            config, list(raw["seeds"]), args.output, **raw.get("experiment", {})
        )
    else:
        manifest = save_run(config, args.output, **raw.get("experiment", {}))
    print(manifest["test_metrics"])


if __name__ == "__main__":
    main()
