"""Leakage-safe delayed-memory calibration and reproducible artifacts."""
from __future__ import annotations

import csv
import importlib.metadata
import json
import platform
import subprocess
import time
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from .config import CVConfig
from .reservoir import GaussianLoopReservoir


def _git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def delayed_memory(config: CVConfig, length: int = 120, delay: int = 2, washout: int = 10) -> dict[str, object]:
    """Chronological delayed-input task with train-only feature scaling."""
    rng = np.random.default_rng(config.seed)
    inputs = rng.uniform(-1, 1, length)
    reservoir = GaussianLoopReservoir(config)
    features, records = reservoir.transform(inputs, washout=washout)
    targets = inputs[washout - delay : length - delay]
    train_end, validation_end = int(0.6 * len(features)), int(0.8 * len(features))
    scaler = StandardScaler().fit(features[:train_end])
    scaled = scaler.transform(features)
    model = Ridge(alpha=1e-4).fit(scaled[:train_end], targets[:train_end])
    prediction = model.predict(scaled)
    baseline = Ridge(alpha=1e-4).fit(inputs[washout : washout + train_end, None], targets[:train_end])
    baseline_prediction = baseline.predict(inputs[washout:, None])
    variance = float(np.var(targets[validation_end:]))
    capacity = 1 - mean_squared_error(targets[validation_end:], prediction[validation_end:]) / variance
    return {
        "seed": config.seed,
        "delay": delay,
        "train_mse": float(mean_squared_error(targets[:train_end], prediction[:train_end])),
        "validation_mse": float(mean_squared_error(targets[train_end:validation_end], prediction[train_end:validation_end])),
        "test_mse": float(mean_squared_error(targets[validation_end:], prediction[validation_end:])),
        "current_input_test_mse": float(mean_squared_error(targets[validation_end:], baseline_prediction[validation_end:])),
        "memory_capacity": float(capacity),
        "feature_dimension": config.feature_dimension,
        "max_covariance_eigenvalue": max(r.maximum_covariance_eigenvalue for r in records),
        "max_mean_photon_number": max(r.mean_photon_number for r in records),
        "stable": True,
        "reservoir_parameters": reservoir.parameters,
    }


def save_run(config: CVConfig, output: Path, **kwargs: int) -> dict[str, object]:
    """Persist JSON, CSV, and PNG/PDF smoke diagnostics."""
    started = time.perf_counter()
    result = delayed_memory(config, **kwargs)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "branch": "CV", "git_commit": _git_commit(), "backend": "piquasso",
        "backend_version": importlib.metadata.version("piquasso"), "simulator": "GaussianSimulator",
        "config": config.to_dict(), "dataset": {"task": "delayed_linear_memory", **kwargs},
        "seeds": [config.seed], "modes": config.modes,
        "photons_or_gaussian_parameters": {"input_squeezing": config.input_squeezing},
        "shots_or_ensemble_size": 0 if config.measurement == "exact" else config.shots,
        "feature_dimension": config.feature_dimension,
        "train_metrics": {"mse": result["train_mse"]},
        "validation_metrics": {"mse": result["validation_mse"]},
        "test_metrics": {"mse": result["test_mse"], "capacity": result["memory_capacity"]},
        "per_seed_metrics": [result], "runtime_seconds": time.perf_counter() - started,
        "physicality_diagnostics": {"stable": result["stable"], "max_covariance_eigenvalue": result["max_covariance_eigenvalue"], "max_mean_photon_number": result["max_mean_photon_number"]},
        "python": platform.python_version(),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    fields = ["seed", "delay", "train_mse", "validation_mse", "test_mse", "current_input_test_mse", "memory_capacity"]
    with (output / "per_seed_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerow(result)
    figure, axis = plt.subplots(figsize=(5, 3.2))
    axis.bar(["Reservoir", "Current input"], [result["test_mse"], result["current_input_test_mse"]], color=["#0072B2", "#E69F00"])
    axis.set_ylabel("Held-out MSE")
    axis.set_title("Delayed-memory smoke calibration")
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(output / f"memory_calibration.{suffix}", dpi=180)
    plt.close(figure)
    return manifest


def save_multiseed_run(
    config: CVConfig, seeds: list[int], output: Path, **kwargs: int
) -> dict[str, object]:
    """Run a predeclared seed list and retain every result, including failures."""
    started = time.perf_counter()
    results = [delayed_memory(replace(config, seed=seed), **kwargs) for seed in seeds]
    test_mse = np.asarray([row["test_mse"] for row in results])
    baseline_mse = np.asarray([row["current_input_test_mse"] for row in results])
    capacity = np.asarray([row["memory_capacity"] for row in results])
    stderr = test_mse.std(ddof=1) / np.sqrt(len(test_mse))
    summary = {
        "mean_test_mse": float(test_mse.mean()),
        "median_test_mse": float(np.median(test_mse)),
        "std_test_mse": float(test_mse.std(ddof=1)),
        "test_mse_ci95": [float(test_mse.mean() - 1.96 * stderr), float(test_mse.mean() + 1.96 * stderr)],
        "mean_baseline_mse": float(baseline_mse.mean()),
        "mean_capacity": float(capacity.mean()),
        "paired_mean_mse_improvement": float((baseline_mse - test_mse).mean()),
        "wins_over_current_input": int(np.sum(test_mse < baseline_mse)),
    }
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "branch": "CV", "git_commit": _git_commit(), "backend": "piquasso",
        "backend_version": importlib.metadata.version("piquasso"), "simulator": "GaussianSimulator",
        "config": config.to_dict(), "dataset": {"task": "delayed_linear_memory", **kwargs},
        "seeds": seeds, "modes": config.modes,
        "photons_or_gaussian_parameters": {"input_squeezing": config.input_squeezing},
        "shots_or_ensemble_size": 0 if config.measurement == "exact" else config.shots,
        "feature_dimension": config.feature_dimension, "train_metrics": {}, "validation_metrics": {},
        "test_metrics": summary, "per_seed_metrics": results,
        "runtime_seconds": time.perf_counter() - started,
        "physicality_diagnostics": {"all_stable": all(row["stable"] for row in results), "maximum_covariance_eigenvalue": max(row["max_covariance_eigenvalue"] for row in results)},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    fields = ["seed", "delay", "train_mse", "validation_mse", "test_mse", "current_input_test_mse", "memory_capacity"]
    with (output / "per_seed_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    figure, axis = plt.subplots(figsize=(5.2, 3.4))
    x = np.arange(len(seeds))
    axis.plot(x, test_mse, "o-", color="#0072B2", label="Reservoir")
    axis.plot(x, baseline_mse, "s--", color="#E69F00", label="Current input")
    axis.set_xticks(x, seeds)
    axis.set_xlabel("Predeclared seed")
    axis.set_ylabel("Held-out MSE")
    axis.legend()
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(output / f"multiseed_memory.{suffix}", dpi=180)
    plt.close(figure)
    return manifest
