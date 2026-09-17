"""Washout, scaling and qubit contraction controls; no test-based model selection."""

import json
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np

from cv_mb_qrc.reservoirs import CVConfig, CVMBReservoir, GraphixMBReservoir
from cv_mb_qrc.reservoirs.diagnostics import contraction, feature_diagnostics
from cv_mb_qrc.reservoirs.results import atomic_json


def run_supplementary(output):
    from main import evaluate

    output = Path(output)
    config = json.loads((output / "raw/config.json").read_text())
    u = np.asarray(json.loads((output / "raw/datasets.json").read_text())["iid"])
    indices = {
        k: np.array(v)
        for k, v in json.loads((output / "raw/split_indices.json").read_text()).items()
    }
    washout_rows, scaling = [], []
    for seed in config["seeds"]:
        for washout in (max(10, config["washout"] // 2), config["washout"], 2 * config["washout"]):

            def factory():
                return CVMBReservoir(CVConfig(seed=seed))

            row = evaluate(
                "cv_B", seed, "iid", u, indices, {**config, "washout": washout}, factory=factory
            )
            washout_rows.append(
                {
                    "seed": seed,
                    "washout": washout,
                    "scores": row["scores"],
                    "splits": {k: v["indices"] for k, v in row["splits"].items()},
                }
            )
        for modes in (1, 2, 4):
            start = perf_counter()
            model = CVMBReservoir(replace(CVConfig(seed=seed), memory_modes=modes))
            compiled = perf_counter() - start
            result = model.run_sequence(u[:100])
            scaling.append(
                {
                    "seed": seed,
                    "memory_modes": modes,
                    "steps": 100,
                    "feature_dimension": len(result.feature_names),
                    "construction_seconds": compiled,
                    "execution_seconds": result.seconds,
                    "resources": result.resources,
                    "diagnostics": feature_diagnostics(result.features),
                }
            )
    atomic_json(output / "raw/washout.json", washout_rows)
    atomic_json(output / "raw/scaling.json", scaling)
    states = [np.diag([1.0, 0.0]), np.diag([0.0, 1.0]), np.ones((2, 2)) / 2]
    atomic_json(
        output / "raw/qubit_contraction.json", contraction(GraphixMBReservoir, u[:60], states)
    )


if __name__ == "__main__":
    run_supplementary(Path(__file__).parent / "results")
