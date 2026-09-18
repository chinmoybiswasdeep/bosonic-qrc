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

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from .capacity import squared_correlation, test_r2
from .config import CVConfig
from .reservoir import GaussianLoopReservoir


def _git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def delayed_memory(
    config: CVConfig,
    length: int = 120,
    delay: int = 2,
    washout: int = 10,
    data_seed: int = 101,
) -> dict[str, object]:
    """Chronological delayed-input task with independent data seeding."""
    inputs = np.random.default_rng(data_seed).uniform(-1, 1, length)
    reservoir = GaussianLoopReservoir(config)
    features, records = reservoir.transform(inputs, washout=washout)
    targets = inputs[washout - delay : length - delay]
    train_end, validation_end = int(0.6 * len(features)), int(0.8 * len(features))
    scaler = StandardScaler().fit(features[:train_end])
    scaled = scaler.transform(features)
    model = Ridge(alpha=1e-4).fit(scaled[:train_end], targets[:train_end])
    prediction = model.predict(scaled)
    baseline = Ridge(alpha=1e-4).fit(
        inputs[washout : washout + train_end, None], targets[:train_end]
    )
    baseline_prediction = baseline.predict(inputs[washout:, None])
    held_target, held_prediction = targets[validation_end:], prediction[validation_end:]
    return {
        "reservoir_seed": config.reservoir_seed,
        "data_seed": data_seed,
        "delay": delay,
        "train_mse": float(
            mean_squared_error(targets[:train_end], prediction[:train_end])
        ),
        "validation_mse": float(
            mean_squared_error(
                targets[train_end:validation_end], prediction[train_end:validation_end]
            )
        ),
        "test_mse": float(mean_squared_error(held_target, held_prediction)),
        "current_input_test_mse": float(
            mean_squared_error(held_target, baseline_prediction[validation_end:])
        ),
        "squared_correlation_capacity": squared_correlation(
            held_target, held_prediction
        ),
        "test_r2": test_r2(held_target, held_prediction),
        "feature_dimension": config.feature_dimension,
        "max_covariance_eigenvalue": max(
            record.maximum_covariance_eigenvalue for record in records
        ),
        "max_mean_photon_number": max(record.mean_photon_number for record in records),
        "stable": True,
        "reservoir_parameters": reservoir.parameters,
    }


def _seed_manifest(config: CVConfig, data_seeds: list[int], reservoir_seeds: list[int]):
    return {
        "reservoir": reservoir_seeds,
        "data": data_seeds,
        "measurement": [config.measurement_seed],
    }


def save_run(config: CVConfig, output: Path, **kwargs: int) -> dict[str, object]:
    started = time.perf_counter()
    result = delayed_memory(config, **kwargs)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "branch": "CV",
        "git_commit": _git_commit(),
        "backend": "piquasso",
        "backend_version": importlib.metadata.version("piquasso"),
        "simulator": "GaussianSimulator",
        "config": config.to_dict(),
        "dataset": {"task": "delayed_linear_memory", **kwargs},
        "seeds": _seed_manifest(config, [result["data_seed"]], [config.reservoir_seed]),
        "modes": config.modes,
        "photons_or_gaussian_parameters": {"input_squeezing": config.input_squeezing},
        "shots_or_ensemble_size": 0 if config.measurement == "exact" else config.shots,
        "feature_dimension": config.feature_dimension,
        "train_metrics": {"mse": result["train_mse"]},
        "validation_metrics": {"mse": result["validation_mse"]},
        "test_metrics": {
            "mse": result["test_mse"],
            "squared_correlation_capacity": result["squared_correlation_capacity"],
            "test_r2": result["test_r2"],
        },
        "per_seed_metrics": [result],
        "runtime_seconds": time.perf_counter() - started,
        "physicality_diagnostics": {
            "stable": result["stable"],
            "max_covariance_eigenvalue": result["max_covariance_eigenvalue"],
            "max_mean_photon_number": result["max_mean_photon_number"],
        },
        "python": platform.python_version(),
    }
    _write_memory_artifacts(output, manifest, [result])
    return manifest


def save_multiseed_run(
    config: CVConfig, seeds: list[int], output: Path, **kwargs: int
) -> dict[str, object]:
    started = time.perf_counter()
    results = [
        delayed_memory(replace(config, reservoir_seed=seed), **kwargs) for seed in seeds
    ]
    test_mse = np.asarray([row["test_mse"] for row in results])
    baseline_mse = np.asarray([row["current_input_test_mse"] for row in results])
    capacities = np.asarray([row["squared_correlation_capacity"] for row in results])
    stderr = test_mse.std(ddof=1) / np.sqrt(len(test_mse))
    summary = {
        "mean_test_mse": float(test_mse.mean()),
        "median_test_mse": float(np.median(test_mse)),
        "std_test_mse": float(test_mse.std(ddof=1)),
        "test_mse_ci95": [
            float(test_mse.mean() - 1.96 * stderr),
            float(test_mse.mean() + 1.96 * stderr),
        ],
        "mean_baseline_mse": float(baseline_mse.mean()),
        "mean_squared_correlation_capacity": float(capacities.mean()),
        "paired_mean_mse_improvement": float((baseline_mse - test_mse).mean()),
        "wins_over_current_input": int(np.sum(test_mse < baseline_mse)),
    }
    output.mkdir(parents=True, exist_ok=True)
    data_seed = int(kwargs.get("data_seed", 101))
    manifest = {
        "branch": "CV",
        "git_commit": _git_commit(),
        "backend": "piquasso",
        "backend_version": importlib.metadata.version("piquasso"),
        "simulator": "GaussianSimulator",
        "config": config.to_dict(),
        "dataset": {"task": "delayed_linear_memory", **kwargs},
        "seeds": _seed_manifest(config, [data_seed], seeds),
        "modes": config.modes,
        "feature_dimension": config.feature_dimension,
        "test_metrics": summary,
        "per_seed_metrics": results,
        "runtime_seconds": time.perf_counter() - started,
        "physicality_diagnostics": {
            "all_stable": all(row["stable"] for row in results),
            "maximum_covariance_eigenvalue": max(
                row["max_covariance_eigenvalue"] for row in results
            ),
        },
    }
    _write_memory_artifacts(output, manifest, results)
    return manifest


def _write_memory_artifacts(
    output: Path, manifest: dict[str, object], results: list[dict[str, object]]
) -> None:
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    fields = [
        "reservoir_seed",
        "data_seed",
        "delay",
        "train_mse",
        "validation_mse",
        "test_mse",
        "current_input_test_mse",
        "squared_correlation_capacity",
        "test_r2",
    ]
    with (output / "per_seed_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
