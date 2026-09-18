"""Versioned, resumable experiment and reporting infrastructure."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SCHEMA_VERSION = "2.0.0"
GATE_STATUSES = {
    "PASS_SMOKE",
    "PASS_CALIBRATION",
    "READY_FOR_PRODUCTION",
    "FAIL_NUMERICAL",
    "FAIL_STATISTICAL",
    "FAIL_BACKEND",
    "INCOMPLETE",
}


def canonical_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def file_checksums(directory: Path, exclude: Iterable[str] = ()) -> dict[str, str]:
    excluded = set(exclude)
    checksums: dict[str, str] = {}
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        name = path.relative_to(directory).as_posix()
        if name in excluded:
            continue
        checksums[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return checksums


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def environment_metadata(packages: Iterable[str]) -> dict[str, object]:
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "dependencies": versions,
    }


@dataclass(frozen=True)
class RunKey:
    reservoir_seed: int
    data_seed: int
    measurement_seed: int
    model: str

    def encoded(self) -> str:
        return canonical_hash(self.__dict__)[:20]


class ResultJournal:
    """Append-only JSONL journal with exact resume and duplicate rejection."""

    def __init__(self, directory: Path, config: dict[str, object]):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / "runs.jsonl"
        self.config_hash = canonical_hash(config)
        self.completed: set[str] = set()
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                record = json.loads(line)
                if record["schema_version"] != SCHEMA_VERSION:
                    raise ValueError("incompatible result schema")
                if record["config_hash"] != self.config_hash:
                    raise ValueError("incompatible config hash in resume directory")
                key = record["run_key"]
                if key in self.completed:
                    raise ValueError(f"duplicate run key in journal: {key}")
                self.completed.add(key)

    def has(self, key: RunKey) -> bool:
        return key.encoded() in self.completed

    def append(self, key: RunKey, payload: dict[str, object]) -> None:
        encoded = key.encoded()
        if encoded in self.completed:
            raise ValueError(f"duplicate run rejected: {encoded}")
        record = {
            "schema_version": SCHEMA_VERSION,
            "config_hash": self.config_hash,
            "run_key": encoded,
            "key_fields": key.__dict__,
            "payload": payload,
        }
        with self.path.open("a", encoding="utf-8", newline="") as handle:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.completed.add(encoded)

    def records(self) -> list[dict[str, object]]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines()]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty result table")
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: tuple(str(row.get(k, "")) for k in fields)))


def write_gate(
    directory: Path,
    status: str,
    checks: list[dict[str, object]],
    blocking_reasons: list[str],
    next_action: str,
) -> dict[str, object]:
    if status not in GATE_STATUSES:
        raise ValueError(f"invalid gate status: {status}")
    gate = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "checks": checks,
        "blocking_reasons": blocking_reasons,
        "next_action": next_action,
    }
    (directory / "gate_report.json").write_text(
        json.dumps(gate, indent=2, allow_nan=False), encoding="utf-8"
    )
    lines = [
        f"# Gate: {status}",
        "",
        *[
            f"- {'PASS' if check['passed'] else 'FAIL'}: {check['name']} — {check['detail']}"
            for check in checks
        ],
        "",
        "## Blocking reasons",
        "",
        *([f"- {reason}" for reason in blocking_reasons] or ["- None"]),
        "",
        f"Next action: {next_action}",
    ]
    (directory / "gate_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return gate


def finalize_manifest(
    directory: Path,
    manifest: dict[str, object],
    started_at: datetime,
    packages: Iterable[str],
) -> dict[str, object]:
    finished = datetime.now(timezone.utc)
    manifest.update(
        {
            "schema_version": SCHEMA_VERSION,
            "config_hash": canonical_hash(manifest["resolved_config"]),
            "git_commit": git_commit(),
            "started_at_utc": started_at.isoformat(),
            "finished_at_utc": finished.isoformat(),
            "wall_time_seconds": (finished - started_at).total_seconds(),
            "environment": environment_metadata(packages),
        }
    )
    manifest["artifact_checksums"] = file_checksums(directory, exclude={"manifest.json"})
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8"
    )
    return manifest


def summarize(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=float)
    q1, q3 = np.quantile(array, [0.25, 0.75])
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "standard_deviation": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "iqr": float(q3 - q1),
    }
