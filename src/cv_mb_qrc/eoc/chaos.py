"""Symmetry-resolved closed-system chaos diagnostics independent of tasks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .core import (
    bose_hubbard_parts,
    exact_unitary,
    fixed_number_basis,
    infinite_temperature_otoc,
    number_operator,
    reflection_blocks,
    spectral_form_factor,
    symmetry_resolved_level_statistics,
)


@dataclass(frozen=True)
class ChaosScanConfig:
    modes: int = 5
    particles: int = 3
    hopping: float = 1.0
    interactions: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0)
    interval: float = 0.3
    boundary: str = "open"
    disorder: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0)
    edge_fraction: float = 0.1
    bootstrap_replicates: int = 500
    seed: int = 2026

    def __post_init__(self):
        if len(self.disorder) != self.modes:
            raise ValueError("one onsite energy per closed-system mode required")
        if sorted(self.interactions) != list(self.interactions):
            raise ValueError("interaction scan must be ordered")
        if self.boundary != "open":
            raise ValueError("current symmetry resolver supports open reflection sectors only")


def _bootstrap_mean(values, rng, replicates):
    values = np.asarray(values, float)
    if not len(values):
        return [None, None]
    draws = np.asarray(
        [np.mean(rng.choice(values, len(values), replace=True)) for _ in range(replicates)]
    )
    return np.quantile(draws, [0.025, 0.975]).tolist()


def _crossings(parameters, ratios):
    midpoint = 0.5 * (0.3863 + 0.5307)
    score = np.asarray(ratios) - midpoint
    crossings = []
    for left in range(len(score) - 1):
        if score[left] == 0:
            crossings.append(float(parameters[left]))
        elif score[left] * score[left + 1] < 0:
            fraction = -score[left] / (score[left + 1] - score[left])
            crossings.append(
                float(parameters[left] + fraction * (parameters[left + 1] - parameters[left]))
            )
    return crossings, midpoint


def scan_closed_core(config: ChaosScanConfig) -> dict[str, object]:
    """Locate candidate regular/chaotic crossings before any task evaluation."""
    basis = fixed_number_basis(config.modes, config.particles)
    rng = np.random.default_rng(config.seed)
    rows = []
    spectra = []
    times = np.arange(0, 21, dtype=float) * config.interval
    for interaction in config.interactions:
        parts = bose_hubbard_parts(
            basis,
            config.hopping,
            interaction,
            np.asarray(config.disorder),
            config.boundary,
        )
        blocks = reflection_blocks(parts.total, basis)
        statistics = symmetry_resolved_level_statistics(blocks, config.edge_fraction)
        ratios = np.asarray(statistics["ratios"])
        eigenvalues = np.concatenate([np.linalg.eigvalsh(block) for block in blocks.values()])
        spectra.append(eigenvalues)
        local_left = number_operator(basis, 0)
        local_right = number_operator(basis, config.modes - 1)
        otoc = [
            infinite_temperature_otoc(exact_unitary(parts.total, time), local_left, local_right)
            for time in times
        ]
        rows.append(
            {
                "interaction": float(interaction),
                "mean_spacing_ratio": statistics["mean_ratio"],
                "spacing_ratio_ci95": _bootstrap_mean(ratios, rng, config.bootstrap_replicates),
                "spacing_ratios": statistics["ratios"],
                "sector_ratios": statistics["by_sector"],
                "otoc": otoc,
                "hilbert_dimension": len(basis),
                "sector_dimensions": {name: len(block) for name, block in blocks.items()},
            }
        )
    mean_ratios = [row["mean_spacing_ratio"] for row in rows]
    crossings, midpoint = _crossings(np.asarray(config.interactions), mean_ratios)
    form_factor = spectral_form_factor(spectra, times)
    for row in rows:
        if crossings:
            distance = min(crossings, key=lambda crossing: abs(row["interaction"] - crossing))
            sign = 1 if row["mean_spacing_ratio"] >= midpoint else -1
            row["signed_distance_to_nearest_boundary"] = sign * abs(row["interaction"] - distance)
        else:
            row["signed_distance_to_nearest_boundary"] = None
    return {
        "rows": rows,
        "candidate_boundaries": crossings,
        "classification_midpoint": midpoint,
        "reference_ratios": {"poisson": 0.3863, "goe": 0.5307, "gue": 0.5996},
        "spectral_form_factor": {key: value.tolist() for key, value in form_factor.items()},
        "times": times.tolist(),
        "symmetry_policy": "fixed total N; open-chain reflection sectors separated",
        "edge_policy": f"discard fixed fraction {config.edge_fraction} per sector",
        "boundary_independent_of_performance": True,
    }
