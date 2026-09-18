"""Small, reproducible EOC smoke study; never a publication claim."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from .chaos import ChaosScanConfig, scan_closed_core
from .core import (
    bose_hubbard_parts,
    exact_unitary,
    fixed_number_basis,
    strang_unitary,
    trace_distance,
)
from .fock_loop import FockLoopConfig, NonGaussianCVReservoir


def _hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _memory_score(features, inputs):
    features = np.asarray(features)[2:]
    target = np.asarray(inputs)[1:-1]
    split = max(3, len(target) // 2)
    train, test = np.arange(split), np.arange(split, len(target))
    mean, scale = features[train].mean(0), features[train].std(0)
    keep = scale > 1e-12
    if not np.any(keep) or not len(test):
        return 0.0
    standardized = (features[:, keep] - mean[keep]) / scale[keep]
    coefficients = np.linalg.pinv(standardized[train]) @ (
        target[train] - target[train].mean()
    )
    prediction = standardized[test] @ coefficients + target[train].mean()
    denominator = np.sum((target[test] - target[test].mean()) ** 2)
    return float(max(0.0, 1 - np.sum((target[test] - prediction) ** 2) / denominator))


def _density(model):
    state = model.state
    if hasattr(state, "matrix"):
        return np.asarray(state.matrix)
    return np.asarray(state)


def _open_scan(raw, boundaries):
    rng = np.random.default_rng(raw["seeds"]["data"])
    inputs = rng.uniform(0, 1, raw["open_system"]["input_length"])
    rows = []
    for interaction in raw["open_system"]["interactions"]:
        modes = raw["open_system"]["memory_modes"]
        settings = FockLoopConfig(
            memory_modes=modes,
            cutoff=raw["open_system"]["cutoff"],
            hopping=1.0,
            interaction=float(interaction),
            disorder=(0.0,) * (modes + 1),
            interval=raw["open_system"]["interval"],
            transmissivity=raw["open_system"]["transmissivity"],
            seed=raw["seeds"]["reservoir"],
        )
        first = NonGaussianCVReservoir(settings)
        second = NonGaussianCVReservoir(settings)
        features = []
        distances = []
        for index, value in enumerate(inputs):
            first_value = value
            second_value = 1 - value if index == 0 else value
            features.append(first.step(float(first_value))["features"])
            second.step(float(second_value))
            distances.append(trace_distance(_density(first), _density(second)))
        physical = first.step(float(inputs[-1]))
        diagnostics = physical["diagnostics"]
        nearest = (
            min(boundaries, key=lambda item: abs(interaction - item))
            if boundaries
            else None
        )
        rows.append(
            {
                "interaction": float(interaction),
                "signed_distance_to_boundary": (
                    None if nearest is None else float(interaction - nearest)
                ),
                "delayed_memory_score": _memory_score(features, inputs),
                "final_trace_distance_after_input_perturbation": distances[-1],
                "fading_memory_curve": distances,
                "trace": diagnostics["trace"],
                "minimum_eigenvalue": diagnostics["minimum_eigenvalue"],
                "boundary_shell_population": diagnostics.get(
                    "boundary_shell_population"
                ),
                "retained_input_norm": diagnostics.get("retained_input_norm"),
                "mean_photon_number": diagnostics["mean_photon_number"],
                "maximum_supported_photon_number": diagnostics[
                    "maximum_supported_photon_number"
                ],
                "channel_metric_scope": "empirical trace-distance fading only",
            }
        )
    return rows


def _trotter_convergence(config):
    basis = fixed_number_basis(config.modes, config.particles)
    interaction = config.interactions[len(config.interactions) // 2]
    parts = bose_hubbard_parts(
        basis,
        config.hopping,
        interaction,
        np.asarray(config.disorder),
        config.boundary,
    )
    exact = exact_unitary(parts.total, config.interval)
    errors = {
        str(substeps): float(
            np.linalg.norm(strang_unitary(parts, config.interval, substeps) - exact)
        )
        for substeps in (1, 2, 4)
    }
    return {
        "representative_interaction": interaction,
        "errors": errors,
        "converged_monotonically": errors["4"] < errors["2"] < errors["1"],
    }


def _cutoff_convergence(raw):
    inputs = np.random.default_rng(raw["seeds"]["data"]).uniform(0, 1, 5)
    models = []
    interaction = raw["open_system"]["interactions"][
        len(raw["open_system"]["interactions"]) // 2
    ]
    modes = raw["open_system"]["memory_modes"]
    for cutoff in raw["open_system"]["cutoff_sequence"]:
        model = NonGaussianCVReservoir(
            FockLoopConfig(
                memory_modes=modes,
                cutoff=cutoff,
                interaction=float(interaction),
                disorder=(0.0,) * (modes + 1),
                interval=raw["open_system"]["interval"],
                transmissivity=raw["open_system"]["transmissivity"],
                seed=raw["seeds"]["reservoir"],
            )
        )
        last = None
        for value in inputs:
            last = model.step(float(value))
        models.append((cutoff, model, last))
    return [
        {
            "cutoff": cutoff,
            **previous.compare_state(current),
            "boundary_population": last["diagnostics"]["boundary_shell_population"],
            "retained_norm": last["diagnostics"]["retained_input_norm"],
        }
        for (_, previous, _), (cutoff, current, last) in pairwise(models)
    ]


def _plot(output, closed, opened):
    interactions = [row["interaction"] for row in closed["rows"]]
    ratios = [row["mean_spacing_ratio"] for row in closed["rows"]]
    low = [row["spacing_ratio_ci95"][0] for row in closed["rows"]]
    high = [row["spacing_ratio_ci95"][1] for row in closed["rows"]]
    figure, axis = plt.subplots(figsize=(5.4, 3.6))
    axis.plot(interactions, ratios, "o-", label="symmetry-resolved")
    axis.fill_between(interactions, low, high, alpha=0.2)
    axis.axhline(0.3863, linestyle="--", label="Poisson")
    axis.axhline(0.5307, linestyle=":", label="GOE")
    for boundary in closed["candidate_boundaries"]:
        axis.axvline(boundary, color="black", alpha=0.25)
    axis.set(xlabel="U/J", ylabel="Mean adjacent-gap ratio")
    axis.legend(fontsize=8)
    figure.tight_layout()
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(output / f"level_statistics.{suffix}", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(5.4, 3.6))
    axis.plot(
        [row["signed_distance_to_boundary"] for row in opened],
        [row["delayed_memory_score"] for row in opened],
        "o-",
    )
    axis.set(
        xlabel="Signed parameter distance to nearest physical candidate boundary",
        ylabel="Smoke delayed-memory score",
    )
    figure.tight_layout()
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(output / f"performance_after_boundary.{suffix}", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(5.4, 3.6))
    axis.plot(
        closed["spectral_form_factor"]["times"],
        closed["spectral_form_factor"]["unconnected"],
        label="unconnected",
    )
    axis.plot(
        closed["spectral_form_factor"]["times"],
        closed["spectral_form_factor"]["connected"],
        label="connected",
    )
    axis.set(xlabel="Time", ylabel="Ensemble-averaged SFF")
    axis.legend()
    figure.tight_layout()
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(output / f"spectral_form_factor.{suffix}", dpi=180)
    plt.close(figure)


def _resume_manifest(output: Path, config_hash: str, chunk: dict):
    path = output / "manifest.json"
    if not path.exists():
        return None
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("config_hash") != config_hash or manifest.get("chunk") != chunk:
        raise ValueError("Existing output is incompatible with this config/chunk")
    for name, expected in manifest.get("artifact_checksums", {}).items():
        artifact = output / name
        if (
            not artifact.exists()
            or hashlib.sha256(artifact.read_bytes()).hexdigest() != expected
        ):
            raise ValueError(f"Cannot resume: checksum mismatch for {name}")
    return manifest


def run(
    config_path: Path,
    output: Path,
    *,
    chunk_index: int = 0,
    chunk_count: int = 1,
    resume: bool = False,
):
    started = datetime.now(timezone.utc)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config_hash = _hash(raw)
    interactions = raw["open_system"]["interactions"]
    if (
        chunk_count < 1
        or not 0 <= chunk_index < chunk_count
        or chunk_count > len(interactions)
    ):
        raise ValueError(
            "Require 0 <= chunk-index < chunk-count <= open interaction count"
        )
    chunk = {"index": chunk_index, "count": chunk_count}
    if (
        resume
        and (manifest := _resume_manifest(output, config_hash, chunk)) is not None
    ):
        return manifest
    raw_open = json.loads(json.dumps(raw))
    raw_open["open_system"]["interactions"] = interactions[chunk_index::chunk_count]
    chaos_config = ChaosScanConfig(**raw["closed_system"])
    closed = scan_closed_core(chaos_config)
    convergence = _trotter_convergence(chaos_config)
    cutoff_convergence = _cutoff_convergence(raw)
    opened = _open_scan(raw_open, closed["candidate_boundaries"])
    output.mkdir(parents=True, exist_ok=True)
    with (output / "chaos_scan.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "interaction",
            "mean_spacing_ratio",
            "spacing_ratio_ci95",
            "hilbert_dimension",
            "signed_distance_to_nearest_boundary",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(closed["rows"])
    with (output / "open_scan.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = sorted(
            key
            for key, value in opened[0].items()
            if not isinstance(value, (list, dict))
        )
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(opened)
    _plot(output, closed, opened)
    checks = {
        "physical_boundary_candidates_found": bool(closed["candidate_boundaries"]),
        "two_physical_diagnostics_recorded": bool(
            closed["spectral_form_factor"]
            and all(row["otoc"] for row in closed["rows"])
        ),
        "trotter_refinement_monotonic": convergence["converged_monotonically"],
        "open_states_physical": all(
            abs(row["trace"] - 1) < raw["thresholds"]["trace_error"]
            and row["minimum_eigenvalue"] >= -raw["thresholds"]["positivity"]
            for row in opened
        ),
        "cutoff_guard_passed": bool(cutoff_convergence)
        and all(
            row["boundary_population"] <= raw["thresholds"]["boundary_shell"]
            and row.get("retained_norm", 1.0) >= 1 - raw["thresholds"]["boundary_shell"]
            for row in cutoff_convergence
        ),
    }
    smoke_pass = all(checks.values())
    gate = {
        "status": "PASS_SMOKE" if smoke_pass else "FAIL_NUMERICAL",
        "edge_of_many_body_quantum_chaos_claim": False,
        "narrow_claim": "candidate closed-system chaos crossings and open-channel smoke",
        "checks": checks,
        "blocking_reasons": [
            "size refinement is not complete",
            "cutoff refinement is not complete",
            "production statistics are not complete",
            "performance peak is not established",
        ],
    }
    (output / "gate_report.json").write_text(
        json.dumps(gate, indent=2), encoding="utf-8"
    )
    (output / "gate_report.md").write_text(
        f"# Gate: {gate['status']}\n\n"
        + "\n".join(f"- {name}: {passed}" for name, passed in checks.items())
        + "\n\n## EOC claim\n\nNot established.\n",
        encoding="utf-8",
    )
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    manifest = {
        "schema_version": "1.0.0-eoc",
        "branch": "EOC-CV",
        "base_commit": raw["provenance"]["base_commit"],
        "git_commit": commit,
        "resolved_config": raw,
        "config_hash": config_hash,
        "chunk": chunk,
        "closed_chaos_diagnostics": closed,
        "trotter_convergence": convergence,
        "cutoff_convergence": cutoff_convergence,
        "open_channel_and_performance": opened,
        "gate": gate,
        "started_at_utc": started.isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "dependencies": {
            package: importlib.metadata.version(package)
            for package in ("numpy", "scipy", "matplotlib", "PyYAML", "piquasso")
        },
    }
    checksums = {}
    for path in sorted(output.iterdir()):
        if path.is_file() and path.name != "manifest.json":
            checksums[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest["artifact_checksums"] = checksums
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-index", type=int, default=0)
    parser.add_argument("--chunk-count", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    manifest = run(
        args.config,
        args.output,
        chunk_index=args.chunk_index,
        chunk_count=args.chunk_count,
        resume=args.resume,
    )
    print(
        {
            "status": manifest["gate"]["status"],
            "eoc_claim": manifest["gate"]["edge_of_many_body_quantum_chaos_claim"],
        }
    )


if __name__ == "__main__":
    main()
