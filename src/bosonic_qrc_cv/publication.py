"""Generate publication-facing tables and cautious text from raw manifests."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _latex_escape(value: object) -> str:
    return str(value).replace("_", "\\_")


def generate(inputs: list[Path], output: Path) -> dict[str, object]:
    manifests = [
        json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        for path in inputs
    ]
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    seed_registry = []
    for source, manifest in zip(inputs, manifests):
        config = manifest["resolved_config"]
        aggregate = manifest.get("aggregate", {})
        principal = aggregate.get("total_significant_capacity", aggregate.get("reservoir", {}))
        rows.append(
            {
                "branch": manifest["branch"],
                "profile": config.get("experiment", {}).get("profile", "unknown"),
                "gate_status": manifest["gate_status"],
                "runs": len(manifest.get("runs", [])),
                "targets": manifest["target_bank"]["number_generated"],
                "null_surrogates": manifest["target_bank"]["null_surrogates"],
                "mean_significant_capacity": principal.get("mean", ""),
                "ci95_low": principal.get("bootstrap_95_ci", ["", ""])[0],
                "ci95_high": principal.get("bootstrap_95_ci", ["", ""])[1],
                "git_commit": manifest["git_commit"],
                "config_hash": manifest["config_hash"],
                "source": source.as_posix(),
            }
        )
        seed_registry.append(
            {
                "branch": manifest["branch"],
                "profile": config.get("experiment", {}).get("profile"),
                "seeds": config["seeds"],
            }
        )
    fields = list(rows[0])
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "schema_version": "2.0.0",
        "inputs": rows,
        "publication_ready": all(
            row["gate_status"] == "READY_FOR_PRODUCTION" for row in rows
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    columns = ["branch", "profile", "runs", "targets", "null_surrogates", "gate_status"]
    latex = [
        "\\begin{tabular}{llllll}",
        "\\toprule",
        " & ".join(_latex_escape(name) for name in columns) + " \\\\",
        "\\midrule",
        *[
            " & ".join(_latex_escape(row[name]) for name in columns) + " \\\\"
            for row in rows
        ],
        "\\bottomrule",
        "\\end{tabular}",
    ]
    (output / "main_results.tex").write_text("\n".join(latex) + "\n", encoding="utf-8")
    (output / "seed_registry.json").write_text(
        json.dumps(seed_registry, indent=2), encoding="utf-8"
    )
    (output / "methods.md").write_text(
        "# Methods fragment\n\n"
        "Inputs were IID Uniform[-1,1]. The principal readout was a centered SVD "
        "pseudoinverse and the principal held-out metric was C=max(0,R²). Ridge "
        "regularization was selected on a separate validation split using train-only "
        "scaling. Null targets were generated from independent IID streams and the "
        "complete fit/validation procedure was repeated. P-values used the add-one "
        "formula and Benjamini-Hochberg correction was applied within each declared "
        "target family. Uncertainty used a hierarchical bootstrap over reservoir "
        "seeds and data seeds.\n",
        encoding="utf-8",
    )
    (output / "captions.md").write_text(
        "# Figure captions\n\n"
        "- Capacity panels show raw held-out C=max(0,R²) and FDR-significant capacity.\n"
        "- Rank panels show centered feature singular spectra; numerical thresholds "
        "combine absolute and relative tolerances.\n"
        "- CV quantities are temporal IPC. DV quantities are delay-zero static "
        "nonlinear capacity and are not temporal memory.\n",
        encoding="utf-8",
    )
    blockers = []
    if not summary["publication_ready"]:
        blockers.append("No production manifest with READY_FOR_PRODUCTION was supplied.")
    branches = {row["branch"] for row in rows}
    if "CV" in branches:
        blockers.append("CV task/control/robustness matrix is not yet complete.")
    if "DV" in branches:
        blockers.append("DV finite-shot/noise stress matrix is not yet complete.")
    blockers.append(
        "Exact truncated-Fock tomography is unsupported in this repository."
    )
    (output / "limitations.md").write_text(
        "# Limitations\n\n"
        + "\n".join(f"- {item}" for item in blockers)
        + "\n",
        encoding="utf-8",
    )
    gate = {
        "schema_version": "2.0.0",
        "status": "INCOMPLETE" if blockers else "READY_FOR_PRODUCTION",
        "blocking_reasons": blockers,
        "next_action": "Complete preregistered stress matrices, then launch production.",
    }
    (output / "gate_report.json").write_text(
        json.dumps(gate, indent=2), encoding="utf-8"
    )
    (output / "gate_report.md").write_text(
        f"# Gate: {gate['status']}\n\n"
        + "\n".join(f"- {item}" for item in blockers)
        + f"\n\nNext action: {gate['next_action']}\n",
        encoding="utf-8",
    )
    return gate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(generate(args.inputs, args.output))


if __name__ == "__main__":
    main()
