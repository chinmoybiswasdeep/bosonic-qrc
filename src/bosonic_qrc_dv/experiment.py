"""Small static-QRP XOR calibration with full provenance."""

from __future__ import annotations

import csv
import importlib.metadata
import json
import subprocess
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from .config import DVConfig
from .reservoir import LinearOpticalReservoir


def run_xor(config: DVConfig, output: Path) -> dict[str, object]:
    started = time.perf_counter()
    coordinates = np.asarray([[-1, -1], [-1, 1], [1, -1], [1, 1]], dtype=float)
    labels = np.asarray([0, 1, 1, 0])
    reservoir = LinearOpticalReservoir(config)
    input_state = reservoir.dual_rail_input()
    features = np.asarray(
        [
            reservoir.probabilities(state=input_state, coordinates=tuple(point)).values
            for point in coordinates
        ]
    )
    model = LogisticRegression(C=10, random_state=config.reservoir_seed).fit(features, labels)
    predictions = model.predict(features)
    accuracy = float(accuracy_score(labels, predictions))
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "branch": "DV",
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "backend": "perceval",
        "backend_version": importlib.metadata.version("perceval-quandela"),
        "simulator": "SLOS",
        "config": config.to_dict(),
        "dataset": {"task": "xor", "samples": 4},
        "seeds": {"reservoir": [config.reservoir_seed], "measurement": [config.measurement_seed]},
        "modes": config.modes,
        "photons_or_gaussian_parameters": {"photons": config.photons},
        "shots_or_ensemble_size": 0 if config.measurement == "exact" else config.shots,
        "nominal_feature_dimension": len(reservoir.outcome_index),
        "nonzero_feature_dimension": int(np.any(features > 1e-14, axis=0).sum()),
        "numerical_feature_rank": int(np.linalg.matrix_rank(features - features.mean(axis=0))),
        "train_metrics": {"accuracy": accuracy},
        "validation_metrics": {},
        "test_metrics": {},
        "per_seed_metrics": [{"reservoir_seed": config.reservoir_seed, "train_accuracy": accuracy}],
        "runtime_seconds": time.perf_counter() - started,
        "physicality_diagnostics": {
            "probability_sums": features.sum(axis=1).tolist(),
            "unitary_error": float(
                np.linalg.norm(
                    reservoir.unitary.conj().T @ reservoir.unitary - np.eye(config.modes)
                )
            ),
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with (output / "features.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["x", "y", "label", *reservoir.probabilities().outcomes])
        for point, label, row in zip(coordinates, labels, features):
            writer.writerow([*point, label, *row])
    figure, axis = plt.subplots(figsize=(4, 3.4))
    axis.imshow(features, aspect="auto", cmap="viridis")
    axis.set_xlabel("Ordered PNR feature")
    axis.set_ylabel("XOR input")
    axis.set_title("Perceval SLOS QRP features")
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(output / f"xor_features.{suffix}", dpi=180)
    plt.close(figure)
    return manifest
