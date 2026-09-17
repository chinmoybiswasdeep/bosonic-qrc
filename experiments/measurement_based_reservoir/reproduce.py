"""Regenerate tables and SVG/PDF/PNG figures exclusively from preserved raw runs."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cv_mb_qrc.reservoirs.diagnostics import bootstrap_summary
from cv_mb_qrc.reservoirs.results import atomic_json


def reproduce(output):
    output = Path(output)
    manifest = json.loads((output / "raw/manifest.json").read_text())
    runs = [json.loads((output / f"raw/{key}.json").read_text()) for key in manifest]
    for directory in ("csv", "json", "figures", "tables"):
        (output / directory).mkdir(exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.linewidth": 1.6,
        }
    )
    summaries, flat = {}, []
    for task in sorted({r["task"] for r in runs}):
        for method in sorted({r["method"] for r in runs}):
            group = [r for r in runs if r["task"] == task and r["method"] == method]
            fields = {
                "linear_capacity": [r["linear_capacity"] for r in group],
                "nonlinear_capacity": [r["nonlinear_capacity"] for r in group],
                "seconds": [r["seconds"] for r in group],
                "effective_rank": [r["diagnostics"]["effective_rank"] for r in group],
            }
            for target in group[0]["scores"]:
                fields[target] = [r["scores"][target]["r2"] for r in group]
            summaries[f"{task}/{method}"] = {k: bootstrap_summary(v) for k, v in fields.items()}
            for field, summary in summaries[f"{task}/{method}"].items():
                flat.append(
                    {
                        "task": task,
                        "method": method,
                        "metric": field,
                        "mean": summary["mean"],
                        "std": summary["std"],
                        "median": summary["median"],
                        "ci_low": summary["bootstrap95"][0],
                        "ci_high": summary["bootstrap95"][1],
                    }
                )
    atomic_json(output / "json/summary.json", summaries)
    with (output / "csv/summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    table = [
        "| Task | Method | Metric | Mean | SD | 95% bootstrap CI |",
        "|---|---|---|---:|---:|---|",
    ]
    for row in flat:
        if row["metric"] in ("linear_capacity", "nonlinear_capacity", "narma10", "mackey_glass"):
            table.append(
                f"| {row['task']} | {row['method']} | {row['metric']} | {row['mean']:.4g} | {row['std']:.3g} | [{row['ci_low']:.4g}, {row['ci_high']:.4g}] |"
            )
    (output / "tables/results.md").write_text("\n".join(table), encoding="utf-8")

    def save(fig, name):
        fig.tight_layout()
        for suffix in ("svg", "pdf", "png"):
            fig.savefig(output / f"figures/{name}.{suffix}", bbox_inches="tight", dpi=150)
        plt.close(fig)

    methods = ("cv_B", "cv_A", "graphix", "esn_14", "rff_14", "delay", "cv_window")
    config = json.loads((output / "raw/config.json").read_text())
    fig, ax = plt.subplots(figsize=(7, 4))
    for method in methods:
        group = summaries[f"iid/{method}"]
        delays = range(1, config["delays"] + 1)
        ax.errorbar(
            list(delays),
            [group[f"linear_{d}"]["mean"] for d in delays],
            yerr=[group[f"linear_{d}"]["std"] for d in delays],
            label=method,
            marker="o",
            capsize=2,
        )
    ax.set(xlabel="Delay", ylabel="Raw test R² (negative values retained)")
    ax.legend(ncol=2)
    save(fig, "linear_memory")
    fig, ax = plt.subplots(figsize=(7, 4))
    for method in methods:
        group = summaries[f"iid/{method}"]
        ax.errorbar(
            group["linear_capacity"]["mean"],
            group["nonlinear_capacity"]["mean"],
            xerr=group["linear_capacity"]["std"],
            yerr=group["nonlinear_capacity"]["std"],
            fmt="o",
            label=method,
        )
    ax.set(xlabel="Clipped linear capacity", ylabel="Clipped nonlinear capacity")
    ax.legend()
    save(fig, "memory_nonlinearity")
    for target, task in (("narma10", "iid"), ("mackey_glass", "mg")):
        fig, ax = plt.subplots(figsize=(7, 4))
        means = [summaries[f"{task}/{m}"][target]["mean"] for m in methods]
        errors = [summaries[f"{task}/{m}"][target]["std"] for m in methods]
        ax.bar(methods, means, yerr=errors, capsize=2)
        ax.tick_params(axis="x", rotation=35)
        ax.set(ylabel="Test R²", title=target)
        save(fig, target)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for metric, ax in zip(("linear_capacity", "nonlinear_capacity", "narma10"), axes, strict=True):
        values = config["transmissivities"]
        ax.errorbar(
            values,
            [summaries[f"iid/cv_eta_{v}"][metric]["mean"] for v in values],
            yerr=[summaries[f"iid/cv_eta_{v}"][metric]["std"] for v in values],
            marker="o",
        )
        ax.set(xlabel="Intensity transmission", ylabel=metric)
    save(fig, "physical_parameter_sweep")
    fig, ax = plt.subplots(figsize=(7, 4))
    for method in methods:
        group = [r for r in runs if r["task"] == "iid" and r["method"] == method]
        spectrum = np.array([r["diagnostics"]["covariance_spectrum"] for r in group])
        ax.semilogy(np.maximum(spectrum.mean(axis=0), 1e-30), label=method)
    ax.set(xlabel="Singular component", ylabel="Training covariance eigenvalue")
    ax.legend()
    save(fig, "feature_spectrum")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(methods, [summaries[f"iid/{m}"]["seconds"]["mean"] for m in methods])
    ax.tick_params(axis="x", rotation=35)
    ax.set(ylabel="Seconds including readout selection")
    save(fig, "runtime")
    dynamics = json.loads((output / "raw/dynamics.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for eta, row in dynamics.items():
        axes[0].semilogy(
            np.maximum(np.array(row["contraction"]["distances"]).mean(axis=1), 1e-30), label=eta
        )
        axes[1].semilogy(np.maximum(row["impulse"]["feature_distance"], 1e-30), label=eta)
    axes[0].set(xlabel="Time", ylabel="Initial-state feature distance")
    axes[1].set(xlabel="Time", ylabel="Impulse feature distance")
    axes[0].legend(title="transmission")
    save(fig, "fading_memory")
    shots = json.loads((output / "raw/shots.json").read_text())
    fig, ax = plt.subplots(figsize=(6, 3.5))
    counts = sorted({r["shots"] for r in shots})
    errors = [[r["rms_error"] for r in shots if r["shots"] == n] for n in counts]
    ax.errorbar(counts, np.mean(errors, axis=1), yerr=np.std(errors, axis=1, ddof=1), marker="o")
    ax.set(xlabel="Independent persistent trajectories", ylabel="Feature RMS error versus exact")
    save(fig, "shot_error")
    fig, ax = plt.subplots(figsize=(6, 3.5))
    strengths = config["noise_strengths"]
    ax.errorbar(
        strengths,
        [summaries[f"iid/cv_noise_{v}"]["narma10"]["mean"] for v in strengths],
        yerr=[summaries[f"iid/cv_noise_{v}"]["narma10"]["std"] for v in strengths],
        marker="o",
    )
    ax.set(xlabel="Added homodyne variance", ylabel="NARMA10 test R²")
    save(fig, "noise_robustness")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for method in methods:
        group = summaries[f"iid/{method}"]
        names = [f"quadratic_{d}" for d in range(1, config["delays"] + 1)]
        axes[0].plot(
            range(1, len(names) + 1), [group[n]["mean"] for n in names], "o-", label=method
        )
        cross = [key for key in group if key.startswith("cross_")]
        axes[1].plot(range(len(cross)), [group[n]["mean"] for n in cross], "o-", label=method)
    axes[0].set(xlabel="Delay", ylabel="Degree-two self target test R²")
    axes[1].set(xlabel="Cross-delay pair index", ylabel="Degree-two cross target test R²")
    axes[0].legend(fontsize=7)
    save(fig, "nonlinear_capacity")
    controls = (
        "cv_B",
        "cv_A",
        "cv_no_temporal",
        "cv_zero_coupling",
        "cv_no_feedforward",
        "cv_window",
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(
        controls,
        [summaries[f"iid/{m}"]["narma10"]["mean"] for m in controls],
        yerr=[summaries[f"iid/{m}"]["narma10"]["std"] for m in controls],
        capsize=2,
    )
    ax.set(ylabel="NARMA10 test R²")
    ax.tick_params(axis="x", rotation=25)
    save(fig, "ablations")
    paired = {}
    for a, b in (
        ("cv_B", "esn_14"),
        ("cv_B", "rff_14"),
        ("graphix", "esn_3"),
        ("cv_B", "cv_window"),
    ):
        for target in ("narma10", "linear_capacity", "nonlinear_capacity"):
            values_a = summaries[f"iid/{a}"][target]["values"]
            values_b = summaries[f"iid/{b}"][target]["values"]
            paired[f"{a} minus {b}/{target}"] = bootstrap_summary(np.array(values_a) - values_b)
    atomic_json(output / "json/paired_differences.json", paired)
    reference = json.loads((output / "raw/mentpy.json").read_text())
    fig, ax = plt.subplots(figsize=(6, 3.5))
    labels = ["density_max_error", "state_max_error_up_to_phase", "probability_max_error"]
    ax.bar(["density", "state (phase aligned)", "probabilities"], [reference[k] for k in labels])
    ax.set(ylabel="Graphix–MentPy absolute error", title="Corrected wire common subset")
    save(fig, "graphix_mentpy")
    if (output / "raw/washout.json").exists():
        rows = json.loads((output / "raw/washout.json").read_text())
        values = sorted({r["washout"] for r in rows})
        scores = [[r["scores"]["narma10"]["r2"] for r in rows if r["washout"] == v] for v in values]
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.errorbar(
            values, np.mean(scores, axis=1), yerr=np.std(scores, axis=1, ddof=1), marker="o"
        )
        ax.set(xlabel="Independent split washout", ylabel="NARMA10 test R²")
        save(fig, "washout_study")
        rows = json.loads((output / "raw/scaling.json").read_text())
        values = sorted({r["memory_modes"] for r in rows})
        fig, ax = plt.subplots(figsize=(6, 3.5))
        for field in ("construction_seconds", "execution_seconds"):
            times = [[r[field] for r in rows if r["memory_modes"] == v] for v in values]
            ax.errorbar(
                values,
                np.mean(times, axis=1),
                yerr=np.std(times, axis=1, ddof=1),
                marker="o",
                label=field,
            )
        ax.set(xlabel="Persistent Gaussian memory modes", ylabel="Seconds (100 steps)")
        ax.legend()
        save(fig, "resource_scaling")
        quantum = json.loads((output / "raw/qubit_contraction.json").read_text())
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.semilogy(np.maximum(quantum["trace_distances"], 1e-30))
        ax.set(xlabel="Time", ylabel="Pairwise memory trace distance")
        save(fig, "qubit_contraction")
    if (output / "raw/fock.json").exists():
        study = json.loads((output / "raw/fock.json").read_text())
        rows = study["rows"][1:]
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.semilogy(
            [r["cutoff"] for r in rows], [max(r["max_feature_change"], 1e-30) for r in rows], "o-"
        )
        ax.set(
            xlabel="Exclusive total-photon cutoff",
            ylabel="Selected feature change",
            title="Fixed zero-PNR branch only",
        )
        save(fig, "fock_convergence")
    print(f"Regenerated {len(flat)} summary rows from {len(runs)} raw runs", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results")
    reproduce(parser.parse_args().output)
