"""Resumable delay-zero capacity experiment for Perceval static QRP."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from .capacity import (
    benjamini_hochberg,
    capacity_bound,
    conventional_capacity,
    enumerate_targets,
    evaluate_target,
    fit_pseudoinverse,
    hierarchical_bootstrap,
    independent_null_capacities,
    max_statistic_threshold,
    minimum_null_surrogates,
    rank_diagnostics,
    select_ridge_alpha,
    squared_correlation,
    target_bank_diagnostics,
    test_r2,
)
from .config import DVConfig
from .infrastructure import (
    ResultJournal,
    RunKey,
    finalize_manifest,
    write_csv,
    write_gate,
)
from .reservoir import LinearOpticalReservoir


def _features(reservoir, inputs, scale, bias):
    state = reservoir.dual_rail_input()
    return np.asarray(
        [
            reservoir.probabilities(
                state=state,
                coordinates=(
                    scale[0] * value + bias[0],
                    scale[1] * value + bias[1],
                ),
            ).values
            for value in inputs
        ]
    )


def _safe(value):
    return float(value) if np.isfinite(value) else None


def _run_one(raw, specification, definitions, config, data_seed, model):
    rngs = [
        np.random.default_rng(data_seed + offset)
        for offset in (
            0,
            raw["seeds"]["validation_offset"],
            raw["seeds"]["test_offset"],
        )
    ]
    lengths = [
        specification["train_length"],
        specification["validation_length"],
        specification["test_length"],
    ]
    inputs = [rng.uniform(-1, 1, length) for rng, length in zip(rngs, lengths)]
    unitary = np.eye(config.modes) if model == "identity" else None
    reservoir = LinearOpticalReservoir(config, unitary=unitary)
    scale, bias = tuple(specification["phase_scale"]), tuple(specification["phase_bias"])
    feature_sets = [_features(reservoir, values, scale, bias) for values in inputs]
    nonconstant = np.std(feature_sets[0], axis=0) > specification["svd_absolute_tolerance"]
    reduced = [features[:, nonconstant] for features in feature_sets]
    diagnostics = rank_diagnostics(
        reduced[0],
        specification["svd_relative_tolerance"],
        specification["svd_absolute_tolerance"],
    )
    target_sets = [
        np.asarray([evaluate_target(values, target) for target in definitions]) for values in inputs
    ]
    alpha_grid = tuple(float(value) for value in specification["ridge_alphas"])
    rows, null_rows = [], []
    for index, definition in enumerate(definitions):
        train_target, validation_target, held_target = (targets[index] for targets in target_sets)
        fit = fit_pseudoinverse(
            reduced[0],
            train_target,
            specification["svd_relative_tolerance"],
            specification["svd_absolute_tolerance"],
        )
        prediction = fit.predict(reduced[2])
        alpha, coefficients, mean, standard = select_ridge_alpha(
            reduced[0], train_target, reduced[1], validation_target, alpha_grid
        )
        ridge_prediction = (reduced[2] - mean) / standard @ coefficients + train_target.mean()
        pinv_null, ridge_null = independent_null_capacities(
            reduced[0],
            reduced[1],
            reduced[2],
            definition,
            specification["null_surrogates"],
            raw["seeds"]["surrogate"]
            + 1_000_003 * config.reservoir_seed
            + 10_007 * data_seed
            + index
            + (0 if model == "reservoir" else 101),
            specification["svd_relative_tolerance"],
            alpha_grid,
        )
        capacity = conventional_capacity(held_target, prediction)
        ridge_capacity = conventional_capacity(held_target, ridge_prediction)
        rows.append(
            {
                "reservoir_seed": config.reservoir_seed,
                "data_seed": data_seed,
                "measurement_seed": config.measurement_seed,
                "model": model,
                "task_multi_index": json.dumps(definition.multi_index),
                "degree": definition.degree,
                "maximum_delay": 0,
                "interaction_order": 1,
                "test_r2": test_r2(held_target, prediction),
                "capacity": capacity,
                "squared_correlation": squared_correlation(held_target, prediction),
                "ridge_test_r2": test_r2(held_target, ridge_prediction),
                "ridge_capacity": ridge_capacity,
                "selected_alpha": alpha,
                "p_value": float((1 + np.sum(pinv_null >= capacity)) / (len(pinv_null) + 1)),
                "ridge_p_value": float(
                    (1 + np.sum(ridge_null >= ridge_capacity)) / (len(ridge_null) + 1)
                ),
                "null_95_threshold": float(np.quantile(pinv_null, 0.95)),
            }
        )
        null_rows.append(pinv_null)
    significant = benjamini_hochberg(
        np.asarray([row["p_value"] for row in rows]), specification["fdr_q"]
    )
    max_threshold = max_statistic_threshold(np.asarray(null_rows), specification["fdr_q"])
    for row, passed in zip(rows, significant):
        row["fdr_significant"] = bool(passed)
        row["max_statistic_significant"] = bool(row["capacity"] > max_threshold)
        row["significant_capacity"] = row["capacity"] if passed else 0.0
    raw_total = float(sum(row["capacity"] for row in rows))
    significant_total = float(sum(row["significant_capacity"] for row in rows))
    raw_bound = capacity_bound(
        raw_total, diagnostics.numerical_rank, specification["bound_tolerance"]
    )
    significant_bound = capacity_bound(
        significant_total,
        diagnostics.numerical_rank,
        specification["bound_tolerance"],
    )
    summary = {
        "model": model,
        "reservoir_seed": config.reservoir_seed,
        "data_seed": data_seed,
        "measurement_seed": config.measurement_seed,
        "nominal_dimension": feature_sets[0].shape[1],
        "nonzero_nonconstant_dimension": int(nonconstant.sum()),
        "numerical_rank": diagnostics.numerical_rank,
        "stable_rank": diagnostics.stable_rank,
        "participation_rank": diagnostics.participation_rank,
        "condition_number": _safe(diagnostics.condition_number),
        "rank_threshold": diagnostics.threshold,
        "singular_values": diagnostics.singular_values.tolist(),
        "raw_total_capacity": raw_total,
        "significant_total_capacity": significant_total,
        "raw_capacity_bound_passed": raw_bound[0],
        "raw_capacity_bound_residual": raw_bound[1],
        "significant_capacity_bound_passed": significant_bound[0],
        "significant_capacity_bound_residual": significant_bound[1],
        "max_statistic_threshold": max_threshold,
        "target_diagnostics": target_bank_diagnostics(target_sets[2]),
        "backend_calls": int(sum(lengths)),
        "photon_sector_weights": reservoir.probabilities(
            state=reservoir.dual_rail_input()
        ).sector_weights,
    }
    return rows, summary


def _plot(output, rows, summaries):
    figure, axis = plt.subplots(figsize=(5.3, 3.5))
    for model in sorted({row["model"] for row in rows}):
        model_rows = [row for row in rows if row["model"] == model]
        degrees = sorted({row["degree"] for row in model_rows})
        values = [
            np.mean([row["capacity"] for row in model_rows if row["degree"] == degree])
            for degree in degrees
        ]
        significant = [
            np.mean([row["significant_capacity"] for row in model_rows if row["degree"] == degree])
            for degree in degrees
        ]
        axis.plot(degrees, values, "o--", label=f"{model} raw")
        axis.plot(degrees, significant, "o-", label=f"{model} FDR")
    axis.set(xlabel="Instantaneous Legendre degree", ylabel="Mean C=max(0,R²)")
    axis.legend(fontsize=8)
    figure.tight_layout()
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(output / f"static_capacity_by_degree.{suffix}", dpi=180)
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(5.3, 3.5))
    for summary in summaries:
        if summary["singular_values"]:
            axis.semilogy(summary["singular_values"], alpha=0.5, label=summary["model"])
    axis.set(xlabel="Singular-value index", ylabel="Singular value")
    figure.tight_layout()
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(output / f"feature_rank_spectrum.{suffix}", dpi=180)
    plt.close(figure)


def run_static_capacity(
    config_path: Path,
    output: Path,
    chunk_index: int = 0,
    chunk_count: int = 1,
) -> dict[str, object]:
    started_at = datetime.now(timezone.utc)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    specification = raw["capacity"]
    specification.setdefault("svd_relative_tolerance", specification.pop("svd_tolerance", 1e-10))
    specification.setdefault("svd_absolute_tolerance", 1e-12)
    definitions = enumerate_targets(specification["maximum_degree"], 0, 1, False)
    output.mkdir(parents=True, exist_ok=True)
    journal = ResultJournal(output, raw)
    if chunk_count < 1 or not 0 <= chunk_index < chunk_count:
        raise ValueError("require 0 <= chunk_index < chunk_count")
    ordinal = 0
    for model in raw["models"]:
        for reservoir_seed in raw["seeds"]["reservoir"]:
            for data_seed in raw["seeds"]["data"]:
                for measurement_seed in raw["seeds"].get(
                    "measurement", [raw["reservoir"]["measurement_seed"]]
                ):
                    selected = ordinal % chunk_count == chunk_index
                    ordinal += 1
                    if not selected:
                        continue
                    key = RunKey(reservoir_seed, data_seed, measurement_seed, model)
                    if journal.has(key):
                        continue
                    values = {
                        **raw["reservoir"],
                        "reservoir_seed": reservoir_seed,
                        "measurement_seed": measurement_seed,
                    }
                    rows, summary = _run_one(
                        raw,
                        specification,
                        definitions,
                        DVConfig(**values),
                        data_seed,
                        model,
                    )
                    journal.append(key, {"rows": rows, "summary": summary})
    records = journal.records()
    rows = [row for record in records for row in record["payload"]["rows"]]
    summaries = [record["payload"]["summary"] for record in records]
    write_csv(output / "capacities.csv", rows)
    write_csv(
        output / "run_summaries.csv",
        [
            {key: value for key, value in row.items() if not isinstance(value, (list, dict))}
            for row in summaries
        ],
    )
    _plot(output, rows, summaries)
    required = minimum_null_surrogates(len(definitions), specification["fdr_q"])
    enough_nulls = specification["null_surrogates"] >= required
    bounds_pass = all(row["significant_capacity_bound_passed"] for row in summaries)
    target_health = all(
        row["target_diagnostics"]["maximum_absolute_off_diagonal"]
        <= specification.get("maximum_target_correlation", 0.30)
        for row in summaries
    )
    profile = raw.get("experiment", {}).get("profile", "smoke")
    if not bounds_pass or not target_health:
        status = "FAIL_STATISTICAL"
    elif profile == "smoke":
        status = "PASS_SMOKE"
    elif profile == "calibration" and enough_nulls:
        status = "PASS_CALIBRATION"
    elif profile == "production" and enough_nulls and specification["null_surrogates"] >= 4999:
        status = "READY_FOR_PRODUCTION"
    else:
        status = "INCOMPLETE"
    blocking = []
    if not enough_nulls:
        blocking.append(f"null B={specification['null_surrogates']} < required {required}")
    gate = write_gate(
        output,
        status,
        [
            {
                "name": "complete static target family",
                "passed": True,
                "detail": f"{len(definitions)} targets",
            },
            {
                "name": "significant capacity bounds",
                "passed": bounds_pass,
                "detail": "per-run total <= numerical feature rank",
            },
            {
                "name": "FDR resolution",
                "passed": enough_nulls,
                "detail": f"B={specification['null_surrogates']}, required={required}",
            },
            {
                "name": "target orthogonality",
                "passed": target_health,
                "detail": "maximum empirical off-diagonal <= configured threshold",
            },
        ],
        blocking,
        (
            "Run the calibration profile."
            if profile == "smoke"
            else "Review stress tests before production launch."
        ),
    )
    aggregate = {}
    for model in raw["models"]:
        selected = [row for row in summaries if row["model"] == model]
        aggregate[model] = hierarchical_bootstrap(
            np.asarray([row["significant_total_capacity"] for row in selected]),
            np.asarray([row["reservoir_seed"] for row in selected]),
            np.asarray([row["data_seed"] for row in selected]),
            seed=91826,
            replicates=1000,
        )
        aggregate[model]["raw_by_degree"] = {
            str(degree): float(
                np.mean(
                    [
                        row["capacity"]
                        for row in rows
                        if row["model"] == model and row["degree"] == degree
                    ]
                )
            )
            for degree in sorted({row["degree"] for row in rows})
        }
        aggregate[model]["significant_by_degree"] = {
            str(degree): float(
                np.mean(
                    [
                        row["significant_capacity"]
                        for row in rows
                        if row["model"] == model and row["degree"] == degree
                    ]
                )
            )
            for degree in sorted({row["degree"] for row in rows})
        }
    reservoir_runs = {
        (row["reservoir_seed"], row["data_seed"], row["measurement_seed"]): row
        for row in summaries
        if row["model"] == "reservoir"
    }
    identity_runs = {
        (row["reservoir_seed"], row["data_seed"], row["measurement_seed"]): row
        for row in summaries
        if row["model"] == "identity"
    }
    paired_keys = sorted(reservoir_runs.keys() & identity_runs.keys())
    if paired_keys:
        differences = np.asarray(
            [
                reservoir_runs[key]["significant_total_capacity"]
                - identity_runs[key]["significant_total_capacity"]
                for key in paired_keys
            ]
        )
        aggregate["paired_reservoir_minus_identity"] = hierarchical_bootstrap(
            differences,
            np.asarray([key[0] for key in paired_keys]),
            np.asarray([key[1] for key in paired_keys]),
            seed=91827,
            replicates=1000,
        )
    manifest = {
        "branch": "DV",
        "backend": "perceval",
        "simulator": "SLOS",
        "capacity_type": "static delay-zero nonlinear capacity",
        "temporal_recurrence": False,
        "resolved_config": raw,
        "input_distribution": "iid Uniform[-1,1]",
        "target_bank": {
            **specification,
            "number_generated": len(definitions),
            "required_null_surrogates": required,
        },
        "readout_protocol": {
            "principal": "centered SVD pseudoinverse",
            "robustness": "validation-selected ridge",
            "principal_metric": "held-out C=max(0,R2)",
            "secondary_metric": "held-out squared correlation",
            "constant_features": "removed using absolute tolerance",
        },
        "null_protocol": {
            "type": "independently regenerated iid target streams",
            "fit_and_validation_repeated": True,
            "multiple_testing": "BH within each model/run family",
            "max_statistic_reported": True,
        },
        "runs": summaries,
        "aggregate": aggregate,
        "backend_call_count": int(sum(row["backend_calls"] for row in summaries)),
        "gate_status": gate["status"],
    }
    return finalize_manifest(
        output,
        manifest,
        started_at,
        ["perceval-quandela", "numpy", "scikit-learn", "matplotlib", "PyYAML"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-index", type=int, default=0)
    parser.add_argument("--chunk-count", type=int, default=1)
    args = parser.parse_args()
    manifest = run_static_capacity(
        args.config, args.output, args.chunk_index, args.chunk_count
    )
    print(
        {
            "gate_status": manifest["gate_status"],
            "wall_time_seconds": manifest["wall_time_seconds"],
        }
    )


if __name__ == "__main__":
    main()
