"""Small, leakage-safe memory experiment."""
from __future__ import annotations

import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

from .config import CVConfig
from .reservoir import GaussianLoopReservoir


def delayed_memory(config: CVConfig, length: int = 120, delay: int = 2, washout: int = 10) -> dict[str, object]:
    """Chronological delayed-input benchmark with train-only readout fitting."""
    rng = np.random.default_rng(config.seed)
    inputs = rng.uniform(-1, 1, length)
    features = GaussianLoopReservoir(config).transform(inputs, washout=washout)
    targets = inputs[washout - delay : length - delay]
    split = int(0.7 * len(features))
    model = Ridge(alpha=1e-6).fit(features[:split], targets[:split])
    prediction = model.predict(features[split:])
    baseline = Ridge(alpha=1e-6).fit(inputs[washout:][None, :split].T, targets[:split])
    baseline_prediction = baseline.predict(inputs[washout:][split:, None])
    return {
        "task": "delayed_linear_memory",
        "delay": delay,
        "test_mse": float(mean_squared_error(targets[split:], prediction)),
        "current_input_test_mse": float(mean_squared_error(targets[split:], baseline_prediction)),
        "feature_dimension": config.feature_dimension,
        "backend": "piquasso.GaussianSimulator",
    }


def save_run(config: CVConfig, output: Path, **kwargs: int) -> dict[str, object]:
    """Run and persist a lightweight reproducibility manifest."""
    started = time.perf_counter()
    result = delayed_memory(config, **kwargs)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except OSError:
        commit = "unknown"
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"config": config.to_dict(), "result": result, "runtime_seconds": time.perf_counter() - started,
                "python": platform.python_version(), "git_commit": commit, "shots": 0}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
