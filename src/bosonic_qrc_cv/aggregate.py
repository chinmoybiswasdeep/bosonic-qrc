"""Merge deterministic experiment shards after compatibility validation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from .infrastructure import ResultJournal, RunKey


def aggregate(config_path: Path, output: Path, shards: list[Path]) -> int:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    journal = ResultJournal(output, raw)
    records = []
    for shard in sorted(shards, key=lambda path: path.as_posix()):
        path = shard / "runs.jsonl"
        if not path.exists():
            raise FileNotFoundError(path)
        records.extend(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
        )
    expected_hash = journal.config_hash
    seen: set[str] = set()
    for record in sorted(records, key=lambda item: item["run_key"]):
        if record["config_hash"] != expected_hash:
            raise ValueError("incompatible config hash across shards")
        if record["run_key"] in seen:
            raise ValueError(f"duplicate run across shards: {record['run_key']}")
        seen.add(record["run_key"])
        fields = record["key_fields"]
        journal.append(RunKey(**fields), record["payload"])
    model_count = len(raw.get("models", ["cv_qrc"]))
    expected = (
        len(raw["seeds"]["reservoir"])
        * len(raw["seeds"]["data"])
        * len(raw["seeds"].get("measurement", [0]))
        * model_count
    )
    if len(seen) != expected:
        raise ValueError(f"incomplete merge: {len(seen)} of {expected} runs")
    return len(seen)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("shards", type=Path, nargs="+")
    args = parser.parse_args()
    print({"merged_runs": aggregate(args.config, args.output, args.shards)})


if __name__ == "__main__":
    main()

