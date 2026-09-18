"""Resumable temporal IPC experiment for the Piquasso CV reservoir."""

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
from .config import CVConfig
from .infrastructure import (
    ResultJournal,
    RunKey,
    finalize_manifest,
    write_csv,
    write_gate,
)
from .reservoir import GaussianLoopReservoir


def _sequence(
    config: CVConfig, data_seed: int, samples: int, washout: int, delay: int
) -> tuple[np.ndarray, np.ndarray, int]:
    inputs = np.random.default_rng(data_seed).uniform(-1, 1, washout + delay + samples)
    reservoir = GaussianLoopReservoir(config)
    features, _ = reservoir.transform(inputs, washout=washout + delay)
    return inputs, features, reservoir.execution_count


def _targets(inputs, definitions, washout):
    return np.asarray(
        [evaluate_target(inputs, target)[washout:] for target in definitions]
    )


def _safe(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _run_one(raw, specification, definitions, config, data_seed):
    streams = []
    calls = 0
    for offset, length in (
        (0, specification["train_length"]),
        (raw["seeds"]["validation_offset"], specification["validation_length"]),
        (raw["seeds"]["test_offset"], specification["test_length"]),
    ):
        inputs, features, count = _sequence(
            config,
            data_seed + offset,
            length,
            specification["washout"],
            specification["maximum_delay"],
        )
        streams.append((inputs, features))
        calls += count
    train_features = streams[0][1]
    validation_features = streams[1][1]
    test_features = streams[2][1]
    target_arrays = [
        _targets(inputs, definitions, specification["washout"]) for inputs, _ in streams
    ]
    diagnostics = rank_diagnostics(
        train_features,
        specification["svd_relative_tolerance"],
        specification["svd_absolute_tolerance"],
    )
    alpha_grid = tuple(float(value) for value in specification["ridge_alphas"])
    run_rows = []
    null_rows = []
    for index, definition in enumerate(definitions):
        train_target = target_arrays[0][index]
        validation_target = target_arrays[1][index]
        held_target = target_arrays[2][index]
        fit = fit_pseudoinverse(
            train_features,
            train_target,
            specification["svd_relative_tolerance"],
            specification["svd_absolute_tolerance"],
        )
        prediction = fit.predict(test_features)
        alpha, coefficients, mean, scale = select_ridge_alpha(
            train_features,
            train_target,
            validation_features,
            validation_target,
            alpha_grid,
        )
        ridge_prediction = (
            test_features - mean
        ) / scale @ coefficients + train_target.mean()
        pinv_null, ridge_null = independent_null_capacities(
            train_features,
            validation_features,
            test_features,
            definition,
            specification["null_surrogates"],
            raw["seeds"]["surrogate"]
            + 1_000_003 * config.reservoir_seed
            + 10_007 * data_seed
            + index,
            specification["svd_relative_tolerance"],
            alpha_grid,
        )
        capacity = conventional_capacity(held_target, prediction)
        ridge_capacity = conventional_capacity(held_target, ridge_prediction)
        run_rows.append(
            {
                "reservoir_seed": config.reservoir_seed,
                "data_seed": data_seed,
                "measurement_seed": config.measurement_seed,
                "model": "cv_qrc",
                "task_multi_index": json.dumps(definition.multi_index),
                "degree": definition.degree,
                "maximum_delay": definition.maximum_delay,
                "interaction_order": definition.interaction_order,
                "test_r2": test_r2(held_target, prediction),
                "capacity": capacity,
                "squared_correlation": squared_correlation(held_target, prediction),
                "ridge_test_r2": test_r2(held_target, ridge_prediction),
                "ridge_capacity": ridge_capacity,
                "selected_alpha": alpha,
                "target_variance": float(np.var(held_target)),
                "p_value": float(
                    (1 + np.sum(pinv_null >= capacity)) / (len(pinv_null) + 1)
                ),
                "ridge_p_value": float(
                    (1 + np.sum(ridge_null >= ridge_capacity)) / (len(ridge_null) + 1)
                ),
                "null_95_threshold": float(np.quantile(pinv_null, 0.95)),
            }
        )
        null_rows.append(pinv_null)
    significant = benjamini_hochberg(
        np.asarray([row["p_value"] for row in run_rows]),
        specification["fdr_q"],
    )
    max_threshold = max_statistic_threshold(
        np.asarray(null_rows), specification["fdr_q"]
    )
    for row, passed in zip(run_rows, significant):
        row["fdr_significant"] = bool(passed)
        row["max_statistic_significant"] = bool(row["capacity"] > max_threshold)
        row["significant_capacity"] = row["capacity"] if passed else 0.0
    raw_total = float(sum(row["capacity"] for row in run_rows))
    significant_total = float(sum(row["significant_capacity"] for row in run_rows))
    raw_bound = capacity_bound(
        raw_total, diagnostics.numerical_rank, specification["bound_tolerance"]
    )
    significant_bound = capacity_bound(
        significant_total,
        diagnostics.numerical_rank,
        specification["bound_tolerance"],
    )
    summary = {
        "reservoir_seed": config.reservoir_seed,
        "data_seed": data_seed,
        "measurement_seed": config.measurement_seed,
        "raw_total_capacity": raw_total,
        "significant_total_capacity": significant_total,
        "numerical_rank": diagnostics.numerical_rank,
        "stable_rank": diagnostics.stable_rank,
        "participation_rank": diagnostics.participation_rank,
        "condition_number": _safe(diagnostics.condition_number),
        "rank_threshold": diagnostics.threshold,
        "raw_capacity_bound_passed": raw_bound[0],
        "raw_capacity_bound_residual": raw_bound[1],
        "significant_capacity_bound_passed": significant_bound[0],
        "significant_capacity_bound_residual": significant_bound[1],
        "max_statistic_threshold": max_threshold,
        "target_diagnostics": target_bank_diagnostics(target_arrays[2]),
        "singular_values": diagnostics.singular_values.tolist(),
        "backend_calls": calls,
    }
    return run_rows, summary


def _aggregate(rows, summaries):
    run_values = np.asarray([row["significant_total_capacity"] for row in summaries])
    reservoir_ids = np.asarray([row["reservoir_seed"] for row in summaries])
    data_ids = np.asarray([row["data_seed"] for row in summaries])
    return {
        "total_significant_capacity": hierarchical_bootstrap(
            run_values,
            reservoir_ids,
            data_ids,
            seed=91826,
            replicates=1000,
        ),
        "significant_by_degree": {
            str(degree): float(
                sum(
                    row["significant_capacity"]
                    for row in rows
                    if row["degree"] == degree
                )
                / len(summaries)
            )
            for degree in sorted({row["degree"] for row in rows})
        },
        "raw_by_degree": {
            str(degree): float(
                sum(row["capacity"] for row in rows if row["degree"] == degree)
                / len(summaries)
            )
            for degree in sorted({row["degree"] for row in rows})
        },
        "significant_by_delay": {
            str(delay): float(
                sum(
                    row["significant_capacity"]
                    for row in rows
                    if row["maximum_delay"] == delay
                )
                / len(summaries)
            )
            for delay in sorted({row["maximum_delay"] for row in rows})
        },
        "raw_by_delay": {
            str(delay): float(
                sum(
                    row["capacity"]
                    for row in rows
                    if row["maximum_delay"] == delay
                )
                / len(summaries)
            )
            for delay in sorted({row["maximum_delay"] for row in rows})
        },
    }


def _plot(output, rows, summaries):
    series = [
        (
            "ipc_by_degree",
            "degree",
            "Legendre degree",
            sorted({row["degree"] for row in rows}),
        ),
        (
            "ipc_by_delay",
            "maximum_delay",
            "Maximum delay",
            sorted({row["maximum_delay"] for row in rows}),
        ),
    ]
    for name, field, label, x_values in series:
        raw = [
            np.mean([row["capacity"] for row in rows if row[field] == x])
            for x in x_values
        ]
        significant = [
            np.mean([row["significant_capacity"] for row in rows if row[field] == x])
            for x in x_values
        ]
        figure, axis = plt.subplots(figsize=(5.3, 3.5))
        axis.plot(x_values, raw, "o--", label="raw")
        axis.plot(x_values, significant, "o-", label="FDR significant")
        axis.set(xlabel=label, ylabel="Mean capacity C=max(0,R²)")
        axis.legend()
        figure.tight_layout()
        for suffix in ("png", "pdf", "svg"):
            figure.savefig(output / f"{name}.{suffix}", dpi=180)
        plt.close(figure)
    figure, axis = plt.subplots(figsize=(5.3, 3.5))
    for summary in summaries:
        axis.semilogy(summary["singular_values"], alpha=0.45, color="#0072B2")
        axis.axhline(summary["rank_threshold"], color="#D55E00", alpha=0.25)
    axis.set(xlabel="Singular-value index", ylabel="Singular value")
    figure.tight_layout()
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(output / f"feature_rank_spectrum.{suffix}", dpi=180)
    plt.close(figure)


def run_ipc(
    config_path: Path,
    output: Path,
    chunk_index: int = 0,
    chunk_count: int = 1,
) -> dict[str, object]:
    started_at = datetime.now(timezone.utc)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    specification = raw["capacity"]
    if "maximum_targets" in specification:
        raise ValueError("maximum_targets is forbidden; enumerate complete families")
    specification.setdefault(
        "svd_relative_tolerance", specification.pop("svd_tolerance", 1e-10)
    )
    specification.setdefault("svd_absolute_tolerance", 1e-12)
    definitions = enumerate_targets(
        specification["maximum_degree"],
        specification["maximum_delay"],
        specification["maximum_interaction_order"],
        specification.get("include_cross_delay", True),
    )
    output.mkdir(parents=True, exist_ok=True)
    journal = ResultJournal(output, raw)
    if chunk_count < 1 or not 0 <= chunk_index < chunk_count:
        raise ValueError("require 0 <= chunk_index < chunk_count")
    ordinal = 0
    for reservoir_seed in raw["seeds"]["reservoir"]:
        for data_seed in raw["seeds"]["data"]:
            for measurement_seed in raw["seeds"].get(
                "measurement", [raw["reservoir"]["measurement_seed"]]
            ):
                selected = ordinal % chunk_count == chunk_index
                ordinal += 1
                if not selected:
                    continue
                key = RunKey(reservoir_seed, data_seed, measurement_seed, "cv_qrc")
                if journal.has(key):
                    continue
                config_values = {
                    **raw["reservoir"],
                    "reservoir_seed": reservoir_seed,
                    "measurement_seed": measurement_seed,
                }
                config = CVConfig(**config_values)
                rows, summary = _run_one(
                    raw, specification, definitions, config, data_seed
                )
                journal.append(key, {"rows": rows, "summary": summary})
    records = journal.records()
    rows = [row for record in records for row in record["payload"]["rows"]]
    summaries = [record["payload"]["summary"] for record in records]
    write_csv(output / "capacities.csv", rows)
    write_csv(
        output / "run_summaries.csv",
        [
            {k: v for k, v in row.items() if not isinstance(v, (list, dict))}
            for row in summaries
        ],
    )
    _plot(output, rows, summaries)
    required = minimum_null_surrogates(len(definitions), specification["fdr_q"])
    enough_nulls = specification["null_surrogates"] >= required
    bounds_pass = all(row["significant_capacity_bound_passed"] for row in summaries)
    target_health = all(
        row["target_diagnostics"]["maximum_absolute_off_diagonal"]
        <= specification.get("maximum_target_correlation", 0.25)
        for row in summaries
    )
    profile = raw.get("experiment", {}).get("profile", "smoke")
    if not bounds_pass or not target_health:
        status = "FAIL_STATISTICAL"
    elif profile == "smoke":
        status = "PASS_SMOKE"
    elif profile == "calibration" and enough_nulls:
        status = "PASS_CALIBRATION"
    elif (
        profile == "production"
        and enough_nulls
        and specification["null_surrogates"] >= 4999
    ):
        status = "READY_FOR_PRODUCTION"
    else:
        status = "INCOMPLETE"
    blocking = []
    if not enough_nulls:
        blocking.append(
            f"null B={specification['null_surrogates']} < required {required}"
        )
    checks = [
        {
            "name": "complete target family",
            "passed": True,
            "detail": f"{len(definitions)} targets; no truncation",
        },
        {
            "name": "significant capacity bounds",
            "passed": bounds_pass,
            "detail": "per-run total <= numerical feature rank",
        },
        {
            "name": "FDR resolution",
            "passed": enough_nulls,
            "detail": (
                f"B={specification['null_surrogates']}, required={required}, "
                f"min p={1 / (specification['null_surrogates'] + 1):.6g}"
            ),
        },
        {
            "name": "target orthogonality",
            "passed": target_health,
            "detail": "maximum empirical off-diagonal <= configured threshold",
        },
    ]
    gate = write_gate(
        output,
        status,
        checks,
        blocking,
        (
            "Run the calibration profile."
            if profile == "smoke"
            else "Run the production profile only after reviewing calibration."
        ),
    )
    aggregate = _aggregate(rows, summaries)
    manifest = {
        "branch": "CV",
        "backend": "piquasso",
        "simulator": "GaussianSimulator",
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
            "scaling": "ridge uses train-only standardization",
        },
        "null_protocol": {
            "type": "independently regenerated iid target streams",
            "fit_and_validation_repeated": True,
            "p_value": "(1 + count(null >= observed))/(B+1)",
            "multiple_testing": "BH within each run target family",
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
        ["piquasso", "numpy", "scikit-learn", "matplotlib", "PyYAML"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-index", type=int, default=0)
    parser.add_argument("--chunk-count", type=int, default=1)
    args = parser.parse_args()
    manifest = run_ipc(
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
