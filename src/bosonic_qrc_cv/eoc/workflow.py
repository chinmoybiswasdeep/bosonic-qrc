"""Checkpointed, chunkable EOC workflow shared by branch-specific wrappers."""

from __future__ import annotations

import copy
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import time
import tracemalloc
from itertools import pairwise
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from .core import density_fidelity, trace_distance
from .performance import TaskProtocol, run_task_suite, summarize_against_chaos
from .science import (
    FloquetChaosConfig,
    floquet_unitary,
    quasienergy_statistics,
    scan_floquet_chaos,
)


def _hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def _verify_artifacts(output: Path, manifest):
    checksums = manifest.get("artifact_checksums")
    if not checksums:
        raise ValueError("resume manifest has no artifact checksums")
    for name, expected in checksums.items():
        path = output / name
        if (
            not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest() != expected
        ):
            raise ValueError(f"resume artifact is missing or corrupt: {name}")


def _tuple_config(raw):
    converted = dict(raw)
    converted["sizes"] = tuple(tuple(item) for item in raw["sizes"])
    converted["interactions"] = tuple(raw["interactions"])
    return FloquetChaosConfig(**converted)


def _reservoir_config(config_class, raw, interaction, cutoff=None):
    values = dict(raw)
    values.pop("cutoff_sequence", None)
    values.pop("interactions", None)
    values.pop("diagnostic_input", None)
    values.pop("echo_realizations", None)
    values.pop("echo_length", None)
    values.pop("overmixing_lambda2_threshold", None)
    values.pop("task_seeds", None)
    values.pop("run_tasks", None)
    if cutoff is not None:
        key = "local_cutoff" if "local_cutoff" in values else "cutoff"
        values[key] = cutoff
    values["interaction"] = interaction
    for key in ("disorder", "kick_phases"):
        if key in values:
            values[key] = tuple(values[key])
    return config_class(**values)


def _clone_factory(prototype):
    def factory():
        model = copy.copy(prototype)
        return model.reset()

    return factory


def _align_density(density, basis, union):
    index = {state: position for position, state in enumerate(union)}
    result = np.zeros((len(union), len(union)), complex)
    for row, bra in enumerate(basis):
        for column, ket in enumerate(basis):
            result[index[bra], index[ket]] = density[row, column]
    return result


def _headline(performance):
    if "headline_mean" in performance:
        return performance["headline_mean"]
    tasks = performance["tasks"]
    return {
        "linear_memory_capacity": tasks["linear_memory_capacity"],
        "nonlinear_memory_capacity": tasks["nonlinear_memory_capacity"],
        "narma10_r2": tasks["narma10"]["test_r2"],
        "mackey_glass_r2": tasks["mackey_glass"]["test_r2"],
        "parity_r2": tasks["delayed_parity"]["test_r2"],
        "channel_equalization_r2": tasks["nonlinear_channel_equalization"]["test_r2"],
    }


def cutoff_convergence(
    reservoir_class,
    config_class,
    raw,
    input_domain,
    protocol,
):
    open_raw = raw["open_system"]
    cutoffs = open_raw["cutoff_sequence"]
    interaction = float(open_raw["interactions"][len(open_raw["interactions"]) // 2])
    rng = np.random.default_rng(raw["seeds"]["data"])
    if input_domain == "dv":
        inputs = rng.uniform(0, 1, raw["convergence"]["sequence_length"])
    else:
        inputs = rng.uniform(-1, 1, raw["convergence"]["sequence_length"])
    models = []
    for cutoff in cutoffs:
        model = reservoir_class(
            _reservoir_config(config_class, open_raw, interaction, cutoff=cutoff)
        )
        rows = [model.step(float(value)) for value in inputs]
        models.append((cutoff, model, rows))
    comparisons = []
    for (_, previous, previous_rows), (cutoff, current, current_rows) in pairwise(
        models
    ):
        union = tuple(sorted(set(previous.basis) | set(current.basis)))
        left = _align_density(previous.state, previous.basis, union)
        right = _align_density(current.state, current.basis, union)
        common_features = min(
            len(previous_rows[-1]["features"]),
            len(current_rows[-1]["features"]),
            raw["convergence"]["observable_prefix"],
        )
        comparison = {
            "cutoff": cutoff,
            "trace_distance_to_previous": trace_distance(left, right),
            "fidelity_to_previous": density_fidelity(left, right),
            "common_projection_trace_previous": float(np.trace(left).real),
            "common_projection_trace_current": float(np.trace(right).real),
            "maximum_raw_trace_error": float(
                max(
                    abs(row["diagnostics"]["raw_output_trace"] - 1)
                    for row in previous_rows + current_rows
                )
            ),
            "maximum_boundary_population": float(
                max(
                    row["diagnostics"].get(
                        "local_cutoff_boundary_population",
                        row["diagnostics"]["boundary_shell_population"],
                    )
                    for row in current_rows
                )
            ),
            "observable_linf_difference": float(
                np.max(
                    np.abs(
                        previous_rows[-1]["features"][:common_features]
                        - current_rows[-1]["features"][:common_features]
                    )
                )
            ),
        }
        if raw["convergence"].get("task_score_check", False):
            seed = raw["seeds"]["data"]
            prior_performance = run_task_suite(
                _clone_factory(previous), seed, protocol, input_domain
            )
            current_performance = run_task_suite(
                _clone_factory(current), seed, protocol, input_domain
            )
            prior_headline = _headline(prior_performance)
            current_headline = _headline(current_performance)
            comparison["task_score_differences"] = {
                key: abs(prior_headline[key] - current_headline[key])
                for key in prior_headline
            }
            comparison["maximum_task_score_difference"] = max(
                comparison["task_score_differences"].values()
            )
        comparisons.append(comparison)

    thresholds = raw["convergence"]["thresholds"]
    for comparison in comparisons:
        checks = {
            "trace_distance": comparison["trace_distance_to_previous"]
            <= thresholds["state_trace_distance"],
            "raw_trace": comparison["maximum_raw_trace_error"]
            <= thresholds["raw_trace_error"],
            "boundary_population": comparison["maximum_boundary_population"]
            <= thresholds["boundary_population"],
            "observables": comparison["observable_linf_difference"]
            <= thresholds["observable_difference"],
        }
        if "maximum_task_score_difference" in comparison:
            checks["task_scores"] = (
                comparison["maximum_task_score_difference"]
                <= thresholds["task_score_difference"]
            )
        comparison["checks"] = checks
        comparison["passed"] = all(checks.values())
    return {
        "same_inputs_and_randomness": True,
        "interaction": interaction,
        "cutoffs": cutoffs,
        "comparisons": comparisons,
        "thresholds": thresholds,
        "passed": bool(comparisons) and all(row["passed"] for row in comparisons),
    }


def dv_cutoff_chaos_check(raw):
    if raw["branch"] != "EOC-DV":
        return {"applicable": False}
    closed = _tuple_config(raw["closed_system"])
    modes, particles = closed.sizes[-1]
    rng = np.random.default_rng(closed.seed)
    disorder = rng.normal(size=modes)
    disorder -= disorder.mean()
    disorder *= closed.disorder_strength / np.std(disorder)
    phases = rng.normal(size=modes)
    phases -= phases.mean()
    phases *= closed.phase_kick_strength / np.std(phases)
    interaction = closed.interactions[len(closed.interactions) // 2]
    rows = []
    from .core import fixed_number_basis

    full = fixed_number_basis(modes, particles)
    for cutoff in raw["open_system"]["cutoff_sequence"]:
        basis = tuple(state for state in full if max(state) < cutoff)
        unitary = floquet_unitary(
            basis,
            closed.hopping,
            interaction,
            disorder,
            phases,
            closed.hopping_time,
            closed.interaction_time,
            closed.boundary,
        )
        statistics = quasienergy_statistics(unitary, closed.degeneracy_tolerance)
        rows.append(
            {
                "local_cutoff": cutoff,
                "dimension": len(basis),
                "mean_spacing_ratio": statistics["mean_ratio"],
                "degeneracy_fraction": statistics["degeneracy_fraction"],
            }
        )
    differences = [
        abs(right["mean_spacing_ratio"] - left["mean_spacing_ratio"])
        for left, right in pairwise(rows)
    ]
    threshold = raw["convergence"]["thresholds"]["chaos_statistic_difference"]
    return {
        "applicable": True,
        "rows": rows,
        "differences": differences,
        "threshold": threshold,
        "passed": bool(differences) and differences[-1] <= threshold,
    }


def _coordinate(edge, values, reference_grid):
    estimate = edge["estimate"]
    if estimate is None:
        return {float(value): None for value in values}
    ci = edge["ci95"]
    grid = max(float(np.min(np.diff(reference_grid))), 1e-12)
    scale = grid
    if ci[0] is not None:
        scale = max(grid, 0.5 * (ci[1] - ci[0]))
    return {float(value): float((value - estimate) / scale) for value in values}


def open_scan(
    reservoir_class,
    config_class,
    raw,
    closed,
    input_domain,
    protocol,
    interactions,
):
    open_raw = raw["open_system"]
    coordinate = _coordinate(
        closed["edge_estimate"],
        open_raw["interactions"],
        raw["closed_system"]["interactions"],
    )
    rows = []
    for interaction in interactions:
        started = time.perf_counter()
        prototype = reservoir_class(
            _reservoir_config(config_class, open_raw, float(interaction))
        )
        channel = prototype.channel_diagnostics(open_raw["diagnostic_input"])
        echo_rng = np.random.default_rng(
            raw["seeds"]["reservoir"] + round(1000 * interaction)
        )
        echo_curves = []
        for _ in range(open_raw["echo_realizations"]):
            if input_domain == "dv":
                inputs = echo_rng.uniform(0, 1, open_raw["echo_length"])
            else:
                inputs = echo_rng.uniform(-1, 1, open_raw["echo_length"])
            first = _clone_factory(prototype)()
            second = _clone_factory(prototype)()
            second.state = np.eye(len(second.basis), dtype=complex) / len(second.basis)
            curve = []
            for value in inputs:
                first.step(float(value))
                second.step(float(value))
                curve.append(trace_distance(first.state, second.state))
            echo_curves.append(curve)
        echo_values = np.asarray(echo_curves)
        performances = []
        if open_raw["run_tasks"]:
            for seed in open_raw["task_seeds"]:
                performances.append(
                    run_task_suite(
                        _clone_factory(prototype), int(seed), protocol, input_domain
                    )
                )
        if len(performances) == 1:
            performance = performances[0]
        else:
            headline_samples = {
                key: [_headline(row)[key] for row in performances]
                for key in _headline(performances[0])
            }
            performance = {
                "seeds": performances,
                "headline_mean": {
                    key: float(np.mean(values))
                    for key, values in headline_samples.items()
                },
                "headline_ci95": {
                    key: np.quantile(values, [0.025, 0.975]).tolist()
                    for key, values in headline_samples.items()
                },
                "tasks": performances[0]["tasks"],
            }
        d_eoc = coordinate[float(interaction)]
        if (
            channel["subleading_eigenvalue_modulus"]
            < open_raw["overmixing_lambda2_threshold"]
        ):
            region = "excessively-mixing"
        elif d_eoc is None:
            region = "unresolved"
        elif d_eoc < -0.5:
            region = "regular"
        elif d_eoc <= 0.5:
            region = "crossover"
        else:
            region = "chaotic"
        rows.append(
            {
                "interaction": float(interaction),
                "d_eoc": d_eoc,
                "region": region,
                "channel": channel,
                "echo_state": {
                    "curves": echo_values.tolist(),
                    "mean": np.mean(echo_values, axis=0).tolist(),
                    "ci95": np.quantile(echo_values, [0.025, 0.975], axis=0).T.tolist(),
                    "realizations": len(echo_values),
                    "same_input_per_pair": True,
                },
                "performance": performance,
                "runtime_seconds": time.perf_counter() - started,
            }
        )
    return rows


def _plots(output, closed, opened):
    output.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6.4, 4.0))
    for size in closed["sizes"]:
        x = [point["interaction"] for point in size["points"]]
        y = [point["mean_spacing_ratio"] for point in size["points"]]
        axis.plot(x, y, "o-", label=size["label"])
    edge = closed["edge_estimate"]
    if edge["estimate"] is not None:
        axis.axvline(
            edge["estimate"], color="black", linestyle="--", label="finite-size EOC"
        )
        if edge["ci95"][0] is not None:
            axis.axvspan(*edge["ci95"], alpha=0.15, color="black")
    axis.set(xlabel="U/J", ylabel="circular adjacent-gap ratio")
    axis.legend()
    figure.tight_layout()
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(output / f"finite_size_chaos.{suffix}", dpi=180)
    plt.close(figure)

    if opened and opened[0]["performance"]:
        figure, axis = plt.subplots(figsize=(6.4, 4.0))
        x = [row["d_eoc"] for row in opened]
        for key, label in (
            ("linear_memory_capacity", "linear memory"),
            ("nonlinear_memory_capacity", "nonlinear memory"),
        ):
            y = [_headline(row["performance"])[key] for row in opened]
            axis.plot(x, y, "o-", label=label)
        axis.axvline(0, color="black", linestyle="--")
        axis.set(xlabel=r"$d_{EOC}$", ylabel="test capacity")
        axis.legend()
        figure.tight_layout()
        for suffix in ("png", "pdf", "svg"):
            figure.savefig(output / f"performance_vs_d_eoc.{suffix}", dpi=180)
        plt.close(figure)


def run_workflow(
    config_path,
    output,
    reservoir_class,
    config_class,
    input_domain,
    *,
    chunk_index=0,
    chunk_count=1,
    closed_result=None,
    resume=False,
):
    started = time.perf_counter()
    tracemalloc.start()
    raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    config_hash = _hash(raw)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    closed_path = (
        Path(closed_result) if closed_result else output / "closed_calibration.json"
    )
    if closed_path.exists():
        closed_payload = json.loads(closed_path.read_text(encoding="utf-8"))
        if closed_payload["closed_config_hash"] != _hash(raw["closed_system"]):
            raise ValueError("closed calibration config hash mismatch")
        closed = closed_payload["result"]
    else:
        closed = scan_floquet_chaos(_tuple_config(raw["closed_system"]))
        atomic_json(
            closed_path,
            {
                "closed_config_hash": _hash(raw["closed_system"]),
                "result": closed,
            },
        )
    interactions = raw["open_system"]["interactions"]
    if not 0 <= chunk_index < chunk_count <= len(interactions):
        raise ValueError("require 0 <= chunk-index < chunk-count <= interaction count")
    selected = interactions[chunk_index::chunk_count]
    chunk = {"index": chunk_index, "count": chunk_count, "interactions": selected}
    manifest_path = output / "manifest.json"
    if resume and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["config_hash"] != config_hash or manifest["chunk"] != chunk:
            raise ValueError("resume config/chunk mismatch")
        _verify_artifacts(output, manifest)
        return manifest

    protocol = TaskProtocol(**raw["tasks"])
    cutoff = cutoff_convergence(
        reservoir_class, config_class, raw, input_domain, protocol
    )
    cutoff_chaos = dv_cutoff_chaos_check(raw)
    opened = open_scan(
        reservoir_class,
        config_class,
        raw,
        closed,
        input_domain,
        protocol,
        selected,
    )
    performance_summary = summarize_against_chaos(opened) if opened else {}
    checks = {
        "closed_degeneracy_and_symmetry": closed["validation"] == "PASS",
        "finite_size_edge_with_uncertainty": closed["edge_estimate"]["status"]
        == "finite_size_candidate"
        and closed["edge_estimate"]["ci95"][0] is not None,
        "complete_channels_cptp": all(
            row["channel"]["validation_passed"] for row in opened
        ),
        "channel_spectral_convergence": all(
            row["channel"].get("subleading_eigenvalue_ritz_residual", 0) <= 1e-3
            and row["channel"]["fixed_point_residual"] <= 1e-8
            for row in opened
        ),
        "echo_state_curves_recorded": all(
            row["echo_state"]["realizations"] >= 2 for row in opened
        ),
        "cutoff_convergence": cutoff["passed"],
        "dv_cutoff_chaos": not cutoff_chaos["applicable"] or cutoff_chaos["passed"],
        "all_tasks_connected": all(bool(row["performance"]) for row in opened),
        "four_interacting_memory_modes": raw["open_system"]["memory_modes"] >= 4,
    }
    gate = {
        "status": "PASS_SMOKE" if all(checks.values()) else "FAIL_VALIDATION",
        "checks": checks,
        "edge_of_many_body_quantum_chaos_claim": False,
        "claim_reason": "development workflow only; publication-size paired statistics are not run",
    }
    current_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    manifest = {
        "schema_version": "2.0.0-eoc",
        "branch": raw["branch"],
        "git_commit": current_commit,
        "base_commit": raw["base_commit"],
        "config_hash": config_hash,
        "configuration": raw,
        "chunk": chunk,
        "closed_calibration": closed,
        "cutoff_convergence": cutoff,
        "dv_local_cutoff_chaos": cutoff_chaos,
        "open_system": opened,
        "performance_vs_d_eoc": performance_summary,
        "runtime": {
            "seconds": time.perf_counter() - started,
            "python_tracemalloc_peak_bytes": peak,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "dependencies": {
                package: importlib.metadata.version(package)
                for package in raw["provenance"]["dependencies"]
            },
        },
        "gate": gate,
        "warnings": [
            "Smoke and profiling outputs validate code paths only.",
            "No EOC performance advantage is claimed.",
            "Thouless time is not estimated for small development ensembles.",
        ],
    }
    atomic_json(manifest_path, manifest)
    with (output / "cutoff_convergence.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fields = [
            "cutoff",
            "trace_distance_to_previous",
            "fidelity_to_previous",
            "maximum_raw_trace_error",
            "maximum_boundary_population",
            "observable_linf_difference",
            "maximum_task_score_difference",
            "passed",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in cutoff["comparisons"]:
            writer.writerow({field: row.get(field) for field in fields})
    with (output / "open_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fields = [
            "interaction",
            "d_eoc",
            "region",
            "subleading_eigenvalue_modulus",
            "tp_residual",
            "echo_final_trace_distance",
            "linear_memory_capacity",
            "nonlinear_memory_capacity",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in opened:
            headline = _headline(row["performance"])
            writer.writerow(
                {
                    "interaction": row["interaction"],
                    "d_eoc": row["d_eoc"],
                    "region": row["region"],
                    "subleading_eigenvalue_modulus": row["channel"][
                        "subleading_eigenvalue_modulus"
                    ],
                    "tp_residual": row["channel"]["tp_residual"],
                    "echo_final_trace_distance": row["echo_state"]["mean"][-1],
                    "linear_memory_capacity": headline["linear_memory_capacity"],
                    "nonlinear_memory_capacity": headline["nonlinear_memory_capacity"],
                }
            )
    atomic_json(output / "gate_report.json", gate)
    _plots(output, closed, opened)
    checksums = {}
    for path in output.iterdir():
        if path.is_file() and path.name != "manifest.json":
            checksums[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest["artifact_checksums"] = checksums
    atomic_json(manifest_path, manifest)
    return manifest


def merge_chunks(paths, output):
    manifests = []
    for path in paths:
        source = Path(path)
        manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
        _verify_artifacts(source, manifest)
        manifests.append(manifest)
    if len({row["config_hash"] for row in manifests}) != 1:
        raise ValueError("cannot merge different configurations")
    counts = {row["chunk"]["count"] for row in manifests}
    if len(counts) != 1 or len(manifests) != counts.pop():
        raise ValueError("all expected chunks are required")
    interactions = [
        value for row in manifests for value in row["chunk"]["interactions"]
    ]
    if len(interactions) != len(set(interactions)):
        raise ValueError("duplicate interaction points across chunks")
    expected = set(manifests[0]["configuration"]["open_system"]["interactions"])
    if set(interactions) != expected:
        raise ValueError("merged chunks do not cover the configured interaction grid")
    merged = copy.deepcopy(manifests[0])
    merged["chunk"] = {
        "index": None,
        "count": len(manifests),
        "interactions": sorted(interactions),
        "merged": True,
    }
    merged["open_system"] = sorted(
        [value for row in manifests for value in row["open_system"]],
        key=lambda value: value["interaction"],
    )
    merged["performance_vs_d_eoc"] = summarize_against_chaos(merged["open_system"])
    merged["source_manifests"] = [str(Path(path) / "manifest.json") for path in paths]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "manifest.json", merged)
    return merged
