"""Delay-zero nonlinear processing capacity for static Perceval QRP."""
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
from .config import DVConfig
from .reservoir import LinearOpticalReservoir


def _features(
    reservoir: LinearOpticalReservoir,
    inputs: np.ndarray,
    phase_scale: tuple[float, float],
    phase_bias: tuple[float, float],
) -> np.ndarray:
    state = reservoir.dual_rail_input()
    return np.asarray(
        [
            reservoir.probabilities(
                state=state,
                coordinates=(
                    phase_scale[0] * value + phase_bias[0],
                    phase_scale[1] * value + phase_bias[1],
                ),
            ).values
            for value in inputs
        ]
    )


def run_static_capacity(config_path: Path, output: Path) -> dict[str, object]:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    specification, rows = raw["capacity"], []
    targets = enumerate_targets(
        specification["maximum_degree"], 0, 1, False
    )
    started, backend_calls, run_summaries = time.perf_counter(), 0, []
    alpha_grid = tuple(float(value) for value in specification["ridge_alphas"])
    for model in raw["models"]:
        for reservoir_seed in raw["seeds"]["reservoir"]:
            config = DVConfig(**raw["reservoir"], reservoir_seed=reservoir_seed)
            unitary = np.eye(config.modes) if model == "identity" else None
            reservoir = LinearOpticalReservoir(config, unitary=unitary)
            for data_seed in raw["seeds"]["data"]:
                generators = [
                    np.random.default_rng(data_seed + offset)
                    for offset in (0, raw["seeds"]["validation_offset"], raw["seeds"]["test_offset"])
                ]
                train_inputs = generators[0].uniform(-1, 1, specification["train_length"])
                validation_inputs = generators[1].uniform(-1, 1, specification["validation_length"])
                test_inputs = generators[2].uniform(-1, 1, specification["test_length"])
                phase_scale = tuple(specification["phase_scale"])
                phase_bias = tuple(specification["phase_bias"])
                train_features = _features(reservoir, train_inputs, phase_scale, phase_bias)
                validation_features = _features(reservoir, validation_inputs, phase_scale, phase_bias)
                test_features = _features(reservoir, test_inputs, phase_scale, phase_bias)
                backend_calls += len(train_inputs) + len(validation_inputs) + len(test_inputs)
                nonzero = int(np.any(train_features > 1e-14, axis=0).sum())
                run_rows = []
                for target_index, target in enumerate(targets):
                    train_target = evaluate_target(train_inputs, target)
                    validation_target = evaluate_target(validation_inputs, target)
                    held_target = evaluate_target(test_inputs, target)
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
                            "model": model,
                            "task_multi_index": json.dumps(target.multi_index),
                            "degree": target.degree,
                            "maximum_delay": 0,
                            "interaction_order": 1,
                            "raw_capacity": capacity,
                            "null_threshold": float(np.quantile(null, 0.95)),
                            "p_value": float((1 + np.sum(null >= capacity)) / (len(null) + 1)),
                            "test_r2": test_r2(held_target, prediction),
                            "ridge_test_r2": test_r2(held_target, ridge_prediction),
                            "selected_alpha": selected_alpha,
                            "nominal_feature_dimension": train_features.shape[1],
                            "nonzero_feature_dimension": nonzero,
                            "effective_rank": fit.numerical_rank,
                            "photon_sector_weights": json.dumps(reservoir.probabilities(state=reservoir.dual_rail_input()).sector_weights),
                        }
                    )
                significant = benjamini_hochberg(
                    np.asarray([row["p_value"] for row in run_rows]), specification["fdr_q"]
                )
                for row, passed in zip(run_rows, significant):
                    row["fdr_significant"] = bool(passed)
                    row["significant_capacity"] = row["raw_capacity"] if passed else 0.0
                total = float(sum(row["significant_capacity"] for row in run_rows))
                passed, residual = capacity_bound(
                    total, int(run_rows[0]["effective_rank"]), specification["bound_tolerance"]
                )
                run_summaries.append({"model": model, "reservoir_seed": reservoir_seed, "data_seed": data_seed, "capacity_bound_passed": passed, "capacity_bound_residual": residual, "total_significant_capacity": total, "effective_rank": run_rows[0]["effective_rank"]})
                rows.extend(run_rows)
    gate = "FAIL_CAPACITY_BOUND" if not all(run["capacity_bound_passed"] for run in run_summaries) else "PASS_SMOKE_PIPELINE_INSUFFICIENT_FDR_RESOLUTION"
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "branch": "DV", "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "backend": "perceval", "backend_version": importlib.metadata.version("perceval-quandela"), "simulator": "SLOS",
        "capacity_type": "static delay-zero nonlinear capacity", "temporal_recurrence": False,
        "config": raw, "input_distribution": "iid Uniform[-1,1]", "target_bank": {**specification, "number_generated": len(targets)},
        "seeds": raw["seeds"], "runs": run_summaries,
        "number_of_significant_targets": int(sum(row["fdr_significant"] for row in rows)),
        "runtime_seconds": time.perf_counter() - started, "backend_call_count": backend_calls,
        "gate_status": gate, "minimum_resolvable_p_value": 1 / (specification["null_surrogates"] + 1),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with (output / "capacities.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    figure, axis = plt.subplots(figsize=(5, 3.4))
    for model in raw["models"]:
        model_rows = [row for row in rows if row["model"] == model]
        axis.plot([row["degree"] for row in model_rows], [row["raw_capacity"] for row in model_rows], "o-", label=model)
    axis.set(xlabel="Instantaneous Legendre degree", ylabel="Raw squared-correlation capacity")
    axis.legend()
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(output / f"static_capacity_by_degree.{suffix}", dpi=180)
    plt.close(figure)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = run_static_capacity(args.config, args.output)
    print({"gate_status": manifest["gate_status"], "runtime_seconds": manifest["runtime_seconds"]})


if __name__ == "__main__":
    main()
