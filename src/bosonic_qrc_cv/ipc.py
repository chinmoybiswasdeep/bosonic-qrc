"""Temporal information-processing-capacity calibration for CV QRC."""
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import subprocess
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from .capacity import (
    benjamini_hochberg,
    capacity_bound,
    enumerate_targets,
    evaluate_target,
    fit_pseudoinverse,
    null_capacities,
    select_ridge_alpha,
    squared_correlation,
    test_r2,
)
from .config import CVConfig
from .reservoir import GaussianLoopReservoir


def _sequence(
    config: CVConfig, data_seed: int, samples: int, washout: int, maximum_delay: int
) -> tuple[np.ndarray, np.ndarray, int]:
    inputs = np.random.default_rng(data_seed).uniform(
        -1, 1, washout + maximum_delay + samples
    )
    reservoir = GaussianLoopReservoir(config)
    features, _ = reservoir.transform(inputs, washout=washout + maximum_delay)
    return inputs, features, reservoir.execution_count


def run_ipc(config_path: Path, output: Path) -> dict[str, object]:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    reservoir_raw, specification = raw["reservoir"], raw["capacity"]
    targets = enumerate_targets(
        specification["maximum_degree"],
        specification["maximum_delay"],
        specification["maximum_interaction_order"],
        specification.get("include_cross_delay", True),
        specification.get("maximum_targets"),
    )
    started, rows, backend_calls = time.perf_counter(), [], 0
    spectra: list[float] = []
    target_correlation_maxima = []
    bound_records = []
    alpha_grid = tuple(float(value) for value in specification["ridge_alphas"])
    for reservoir_seed in raw["seeds"]["reservoir"]:
        config = CVConfig(**reservoir_raw, reservoir_seed=reservoir_seed)
        for data_seed in raw["seeds"]["data"]:
            train_inputs, train_features, calls = _sequence(
                config,
                data_seed,
                specification["train_length"],
                specification["washout"],
                specification["maximum_delay"],
            )
            validation_inputs, validation_features, validation_calls = _sequence(
                config,
                data_seed + raw["seeds"]["validation_offset"],
                specification["validation_length"],
                specification["washout"],
                specification["maximum_delay"],
            )
            test_inputs, test_features, test_calls = _sequence(
                config,
                data_seed + raw["seeds"]["test_offset"],
                specification["test_length"],
                specification["washout"],
                specification["maximum_delay"],
            )
            backend_calls += calls + validation_calls + test_calls
            train_targets = [evaluate_target(train_inputs, target)[specification["washout"] :] for target in targets]
            validation_targets = [evaluate_target(validation_inputs, target)[specification["washout"] :] for target in targets]
            test_targets = [evaluate_target(test_inputs, target)[specification["washout"] :] for target in targets]
            correlation = np.corrcoef(np.asarray(test_targets))
            target_correlation_maxima.append(float(np.max(np.abs(correlation - np.eye(len(targets))))))
            run_rows = []
            for target_index, (target, train_target, validation_target, held_target) in enumerate(
                zip(targets, train_targets, validation_targets, test_targets)
            ):
                fit = fit_pseudoinverse(
                    train_features, train_target, specification["svd_tolerance"]
                )
                prediction = fit.predict(test_features)
                capacity = squared_correlation(held_target, prediction)
                null = null_capacities(
                    train_features,
                    test_features,
                    train_target,
                    held_target,
                    specification["null_surrogates"],
                    raw["seeds"]["surrogate"] + target_index,
                    specification["svd_tolerance"],
                )
                selected_alpha, coefficients, mean, scale = select_ridge_alpha(
                    train_features,
                    train_target,
                    validation_features,
                    validation_target,
                    alpha_grid,
                )
                ridge_prediction = (test_features - mean) / scale @ coefficients + train_target.mean()
                run_rows.append(
                    {
                        "reservoir_seed": reservoir_seed,
                        "data_seed": data_seed,
                        "measurement_seed": config.measurement_seed,
                        "model": "cv_qrc",
                        "task_multi_index": json.dumps(target.multi_index),
                        "degree": target.degree,
                        "maximum_delay": target.maximum_delay,
                        "interaction_order": target.interaction_order,
                        "raw_capacity": capacity,
                        "null_threshold": float(np.quantile(null, 0.95)),
                        "p_value": float((1 + np.sum(null >= capacity)) / (len(null) + 1)),
                        "test_r2": test_r2(held_target, prediction),
                        "ridge_test_r2": test_r2(held_target, ridge_prediction),
                        "selected_alpha": selected_alpha,
                        "target_variance": float(np.var(held_target)),
                        "feature_dimension": train_features.shape[1],
                        "effective_rank": fit.numerical_rank,
                        "condition_number": fit.condition_number,
                    }
                )
                spectra.extend(fit.singular_values.tolist())
            significant = benjamini_hochberg(
                np.asarray([row["p_value"] for row in run_rows]), specification["fdr_q"]
            )
            for row, passed in zip(run_rows, significant):
                row["fdr_significant"] = bool(passed)
                row["significant_capacity"] = row["raw_capacity"] if passed else 0.0
            total = float(sum(row["significant_capacity"] for row in run_rows))
            rank = int(run_rows[0]["effective_rank"])
            bound_passed, residual = capacity_bound(total, rank, specification["bound_tolerance"])
            bound_records.append({"reservoir_seed": reservoir_seed, "data_seed": data_seed, "passed": bound_passed, "residual": residual, "rank": rank, "total": total})
            rows.extend(run_rows)
    if not all(record["passed"] for record in bound_records):
        gate_status = "FAIL_CAPACITY_BOUND"
    elif specification["null_surrogates"] < 100:
        gate_status = "PASS_SMOKE_PIPELINE_INSUFFICIENT_FDR_RESOLUTION"
    else:
        gate_status = "PASS"
    output.mkdir(parents=True, exist_ok=True)
    totals_by_degree = {
        str(degree): float(sum(row["significant_capacity"] for row in rows if row["degree"] == degree))
        for degree in sorted({row["degree"] for row in rows})
    }
    manifest = {
        "branch": "CV",
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "backend": "piquasso",
        "backend_version": importlib.metadata.version("piquasso"),
        "simulator": "GaussianSimulator",
        "config": raw,
        "input_distribution": "iid Uniform[-1,1]",
        "target_bank": {**specification, "number_generated": len(targets)},
        "seeds": raw["seeds"],
        "feature_dimension": CVConfig(**reservoir_raw).feature_dimension,
        "number_of_significant_targets": int(sum(row["fdr_significant"] for row in rows)),
        "total_capacity_by_degree": totals_by_degree,
        "total_instantaneous_capacity": float(sum(row["significant_capacity"] for row in rows if row["maximum_delay"] == 0)),
        "total_temporal_capacity": float(sum(row["significant_capacity"] for row in rows if row["maximum_delay"] > 0)),
        "total_cross_delay_capacity": float(sum(row["significant_capacity"] for row in rows if row["interaction_order"] > 1)),
        "capacity_bounds": bound_records,
        "maximum_target_correlation": max(target_correlation_maxima),
        "runtime_seconds": time.perf_counter() - started,
        "backend_call_count": backend_calls,
        "gate_status": gate_status,
        "minimum_resolvable_p_value": 1 / (specification["null_surrogates"] + 1),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with (output / "capacities.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _plot(output, rows, spectra)
    return manifest


def _plot(output: Path, rows: list[dict[str, object]], spectra: list[float]) -> None:
    degree_totals = {}
    for row in rows:
        degree_totals[row["degree"]] = degree_totals.get(row["degree"], 0) + row["significant_capacity"]
    figure, axis = plt.subplots(figsize=(5, 3.4))
    axis.bar(degree_totals.keys(), degree_totals.values(), color="#0072B2")
    axis.set(xlabel="Legendre degree", ylabel="Significant squared-correlation capacity")
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(output / f"ipc_by_degree.{suffix}", dpi=180)
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(5, 3.4))
    axis.semilogy(sorted(spectra, reverse=True), color="#009E73")
    axis.set(xlabel="Singular-value index", ylabel="Singular value")
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(output / f"feature_rank_spectrum.{suffix}", dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = run_ipc(args.config, args.output)
    print({"gate_status": manifest["gate_status"], "runtime_seconds": manifest["runtime_seconds"]})


if __name__ == "__main__":
    main()
