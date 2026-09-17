"""Structured numerical results and portable provenance."""

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np


@lru_cache(maxsize=1)
def environment() -> dict:
    packages: dict[str, str | None] = {}
    for name in (
        "photographiq",
        "cv-mb-qrc",
        "piquasso",
        "graphix",
        "mentpy",
        "numpy",
        "scipy",
        "networkx",
        "scikit-learn",
        "jax",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    roots = {"cv-mb-qrc": Path(__file__).resolve().parents[3]}
    import photographiq

    roots["photographiq"] = Path(photographiq.__file__).resolve().parents[2]
    commits = {}
    dirty = {}
    for name, root in roots.items():
        proc = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        commits[name] = proc.stdout.strip() if proc.returncode == 0 else None
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=False,
        )
        dirty[name] = bool(status.stdout.strip()) if status.returncode == 0 else None
    root = roots["cv-mb-qrc"]
    source_paths = sorted((root / "src/photographiqml/reservoirs").glob("*.py"))
    source_paths += sorted((root / "experiments/measurement_based_reservoir").glob("*.py"))
    hashes = {
        str(p.relative_to(root)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in source_paths
    }
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "commits": commits,
        "dirty_worktrees": dirty,
        "implementation_sha256": hashes,
    }


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, indent=2, allow_nan=False)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=path.name + ".",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    try:
        # Windows scanners/readers can hold a short-lived non-sharing handle.
        for attempt in range(7):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if attempt == 6:
                    raise
                time.sleep(0.02 * 2**attempt)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass
class ReservoirResult:
    features: np.ndarray
    feature_names: tuple[str, ...]
    evolution: str
    estimator: str
    shots: int | None
    outcomes: list = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)
    resources: dict = field(default_factory=dict)
    configuration: dict = field(default_factory=dict)
    environment: dict = field(default_factory=dict)
    seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["features"] = self.features.tolist()
        return data

    def save(self, path):
        atomic_json(path, self.to_dict())

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        data["features"] = np.asarray(data["features"], float)
        data["feature_names"] = tuple(data["feature_names"])
        if not np.isfinite(data["features"]).all():
            raise ValueError("Nonfinite saved features")
        return cls(**data)
