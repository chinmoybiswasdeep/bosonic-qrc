"""Floquet many-body chaos calibration with fixed-parameter ensembles.

This module deliberately keeps physical calibration independent of reservoir
performance.  Disorder breaks spatial symmetries reproducibly; total particle
number remains the only exact sector used by the default workflow.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import log
from typing import Any

import numpy as np
from scipy.linalg import expm

from .core import fixed_number_basis, number_operator, reduced_density_matrix


@dataclass(frozen=True)
class FloquetChaosConfig:
    sizes: tuple[tuple[int, int], ...] = ((4, 3), (5, 3))
    interactions: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0)
    hopping: float = 1.0
    hopping_time: float = 0.37
    interaction_time: float = 0.23
    boundary: str = "open"
    disorder_strength: float = 0.173
    phase_kick_strength: float = 0.117
    realizations: int = 8
    bootstrap_replicates: int = 300
    degeneracy_tolerance: float = 1e-9
    universality_class: str = "COE"
    otoc_steps: int = 8
    seed: int = 2026

    def __post_init__(self):
        if len(self.sizes) < 2:
            raise ValueError("finite-size EOC estimation requires at least two sizes")
        if any(modes < 3 or particles < 1 for modes, particles in self.sizes):
            raise ValueError("each closed size needs >=3 modes and positive particle number")
        if tuple(sorted(self.interactions)) != self.interactions:
            raise ValueError("interaction grid must be sorted")
        if self.realizations < 2 or self.bootstrap_replicates < 20:
            raise ValueError("ensemble and bootstrap sizes are too small")
        if self.universality_class != "COE":
            raise ValueError(
                "this real-hopping protocol has COE symmetry; CUE requires a flux extension"
            )
        if self.boundary not in {"open", "ring"}:
            raise ValueError("boundary must be open or ring")


def _edges(modes: int, boundary: str):
    result = [(site, site + 1) for site in range(modes - 1)]
    if boundary == "ring" and modes > 2:
        result.append((modes - 1, 0))
    return result


def floquet_parts(
    basis: tuple[tuple[int, ...], ...],
    hopping: float,
    interaction: float,
    disorder: np.ndarray,
    phases: np.ndarray,
    boundary: str,
):
    """Return H_J and diagonal H_U in the supplied number sector."""
    modes = len(basis[0])
    if disorder.shape != (modes,) or phases.shape != (modes,):
        raise ValueError("one disorder and kick phase per mode required")
    index = {state: position for position, state in enumerate(basis)}
    h_j = np.zeros((len(basis), len(basis)), complex)
    h_u = np.zeros_like(h_j)
    for column, state in enumerate(basis):
        occupation = np.asarray(state)
        h_j[column, column] = disorder @ occupation
        h_u[column, column] = (
            0.5 * interaction * np.sum(occupation * (occupation - 1)) + phases @ occupation
        )
        for left, right in _edges(modes, boundary):
            for source, destination in ((left, right), (right, left)):
                if state[source] == 0:
                    continue
                changed = list(state)
                changed[source] -= 1
                changed[destination] += 1
                row = index.get(tuple(changed))
                if row is not None:
                    h_j[row, column] += -hopping * np.sqrt(state[source] * (state[destination] + 1))
    if not np.allclose(h_j, h_j.conj().T) or not np.allclose(h_u, h_u.conj().T):
        raise RuntimeError("Floquet generators must be Hermitian")
    return h_j, h_u


def floquet_unitary(
    basis: tuple[tuple[int, ...], ...],
    hopping: float,
    interaction: float,
    disorder: np.ndarray,
    phases: np.ndarray,
    hopping_time: float,
    interaction_time: float,
    boundary: str = "open",
):
    """Exact two-kick stroboscopic model U_F=exp(-iH_U T_U)exp(-iH_J T_J)."""
    h_j, h_u = floquet_parts(basis, hopping, interaction, disorder, phases, boundary)
    return expm(-1j * h_u * interaction_time) @ expm(-1j * h_j * hopping_time)


def quasienergy_statistics(unitary: np.ndarray, tolerance: float):
    phases = np.sort(np.mod(np.angle(np.linalg.eigvals(unitary)), 2 * np.pi))
    spacings = np.diff(np.r_[phases, phases[0] + 2 * np.pi])
    near = spacings <= tolerance
    degeneracy_fraction = float(np.mean(near))
    valid = not bool(np.any(near)) and len(spacings) >= 3
    ratios = np.asarray([], float)
    if valid:
        ratios = np.minimum(spacings, np.roll(spacings, -1)) / np.maximum(
            spacings, np.roll(spacings, -1)
        )
    return {
        "phases": phases,
        "spacings": spacings,
        "ratios": ratios,
        "mean_ratio": float(np.mean(ratios)) if valid else None,
        "degeneracy_fraction": degeneracy_fraction,
        "degeneracy_count": int(np.count_nonzero(near)),
        "degeneracy_tolerance": tolerance,
        "valid": valid,
    }


def random_matrix_sff(times: np.ndarray, dimension: int, universality_class: str):
    tau = np.asarray(times, float) / dimension
    if universality_class == "CUE":
        normalized = np.minimum(tau, 1.0)
    else:
        normalized = np.empty_like(tau)
        early = tau <= 1
        normalized[early] = 2 * tau[early] - tau[early] * np.log1p(2 * tau[early])
        late = ~early
        safe = np.maximum(2 * tau[late] - 1, np.finfo(float).eps)
        normalized[late] = 2 - tau[late] * np.log((2 * tau[late] + 1) / safe)
    return dimension * normalized


def sff_ensemble(unitaries: list[np.ndarray], steps: int, rng, bootstrap_replicates: int, cls: str):
    """SFF ensemble at one fixed physical parameter point."""
    times = np.arange(1, steps + 1, dtype=int)
    traces = np.asarray(
        [
            [np.trace(np.linalg.matrix_power(unitary, int(time))) for time in times]
            for unitary in unitaries
        ]
    )
    samples = np.abs(traces) ** 2
    raw = np.mean(samples, axis=0)
    connected = raw - np.abs(np.mean(traces, axis=0)) ** 2
    boot = np.asarray(
        [
            np.mean(samples[rng.integers(0, len(samples), len(samples))], axis=0)
            for _ in range(bootstrap_replicates)
        ]
    )
    return {
        "times": times.tolist(),
        "raw": raw.tolist(),
        "normalized": (raw / len(unitaries[0])).tolist(),
        "connected": connected.tolist(),
        "ci95": np.quantile(boot, [0.025, 0.975], axis=0).T.tolist(),
        "ensemble_size": len(unitaries),
        "reference_class": cls,
        "random_matrix_reference": random_matrix_sff(times, len(unitaries[0]), cls).tolist(),
        "thouless_time": None,
        "thouless_time_status": "not estimated: development ensembles are insufficient",
    }


def _entropy(density: np.ndarray):
    values = np.linalg.eigvalsh((density + density.conj().T) / 2)
    values = values[values > 1e-14]
    return float(-np.sum(values * np.log(values)))


def eigenvector_diagnostics(unitary: np.ndarray, basis):
    _, vectors = np.linalg.eig(unitary)
    probabilities = np.abs(vectors) ** 2
    ipr = np.sum(probabilities**2, axis=0)
    participation_entropy = -np.sum(
        np.where(probabilities > 0, probabilities * np.log(probabilities), 0), axis=0
    )
    selected = np.linspace(0, len(vectors) - 1, min(8, len(vectors)), dtype=int)
    entropies = []
    keep = tuple(range(max(1, len(basis[0]) // 2)))
    for column in selected:
        vector = vectors[:, column] / np.linalg.norm(vectors[:, column])
        reduced, _ = reduced_density_matrix(np.outer(vector, vector.conj()), basis, keep)
        entropies.append(_entropy(reduced))
    return {
        "mean_ipr": float(np.mean(ipr)),
        "mean_participation_entropy": float(np.mean(participation_entropy)),
        "normalized_participation_entropy": float(np.mean(participation_entropy) / log(len(basis))),
        "mean_bipartite_entanglement_entropy": float(np.mean(entropies)),
        "sampled_eigenstates_for_entanglement": len(entropies),
    }


def floquet_otoc(unitary: np.ndarray, basis, steps: int):
    left = number_operator(basis, 0)
    right = number_operator(basis, len(basis[0]) - 1)
    evolved = left.copy()
    rows = []
    for step in range(steps + 1):
        commutator = evolved @ right - right @ evolved
        rows.append(float(np.real(-np.trace(commutator @ commutator) / len(basis))))
        evolved = unitary.conj().T @ evolved @ unitary
    return rows


def _crossings(parameters, scores, level=0.5):
    result = []
    for index in range(len(parameters) - 1):
        left, right = scores[index] - level, scores[index + 1] - level
        if left == 0:
            result.append(float(parameters[index]))
        elif left * right < 0:
            weight = -left / (right - left)
            result.append(
                float(parameters[index] + weight * (parameters[index + 1] - parameters[index]))
            )
    return result


def _bootstrap_ci(values):
    return np.quantile(np.asarray(values, float), [0.025, 0.975]).tolist()


def _edge_estimate(size_rows: list[dict[str, Any]], config: FloquetChaosConfig, rng):
    poisson = 0.3862943611
    chaotic = 0.5307 if config.universality_class == "COE" else 0.5996
    estimates: list[float] = []
    bootstrap_by_size: list[list[float]] = []
    all_crossings: dict[str, list[float]] = {}
    for size in size_rows:
        means = np.asarray([point["mean_spacing_ratio"] for point in size["points"]])
        scores = np.clip((means - poisson) / (chaotic - poisson), 0, 1)
        crossings = _crossings(np.asarray(config.interactions), scores)
        all_crossings[size["label"]] = crossings
        estimates.append(crossings[0] if crossings else np.nan)
        boot_edges = []
        per_point = [np.asarray(point["realization_mean_ratios"]) for point in size["points"]]
        for _ in range(config.bootstrap_replicates):
            sampled = [
                float(np.mean(values[rng.integers(0, len(values), len(values))]))
                for values in per_point
            ]
            sampled_scores = np.clip((np.asarray(sampled) - poisson) / (chaotic - poisson), 0, 1)
            found = _crossings(np.asarray(config.interactions), sampled_scores)
            if found:
                boot_edges.append(found[0])
        bootstrap_by_size.append(boot_edges)
        size["chaos_scores"] = scores.tolist()
        size["entry_crossings"] = crossings[:1]
        size["additional_crossings"] = crossings[1:]
        size["entry_ci95"] = _bootstrap_ci(boot_edges) if len(boot_edges) >= 10 else [None, None]

    valid: list[tuple[float, float]] = [
        (float(size["dimension"]), edge)
        for size, edge in zip(size_rows, estimates)
        if np.isfinite(edge)
    ]
    if not valid:
        return {
            "status": "unresolved",
            "estimate": None,
            "ci95": [None, None],
            "size_estimates": estimates,
            "all_crossings": all_crossings,
        }
    if len(valid) >= 2:
        inverse_dimensions = np.asarray([1 / item[0] for item in valid])
        finite_edges = np.asarray([item[1] for item in valid])
        estimate = float(np.polyfit(inverse_dimensions, finite_edges, 1)[1])
    else:
        estimate = float(valid[-1][1])
    boot_extrapolated = []
    for replicate in range(config.bootstrap_replicates):
        sample: list[tuple[float, float]] = []
        for size, edges in zip(size_rows, bootstrap_by_size):
            if edges:
                sample.append((float(size["dimension"]), edges[replicate % len(edges)]))
        if len(sample) >= 2:
            boot_extrapolated.append(
                float(
                    np.polyfit([1 / item[0] for item in sample], [item[1] for item in sample], 1)[1]
                )
            )
        elif sample:
            boot_extrapolated.append(float(sample[-1][1]))
    ci = _bootstrap_ci(boot_extrapolated) if len(boot_extrapolated) >= 10 else [None, None]
    return {
        "status": "finite_size_candidate",
        "estimate": estimate,
        "ci95": ci,
        "normalization": "linear extrapolation in inverse Hilbert dimension",
        "size_estimates": estimates,
        "all_crossings": all_crossings,
        "bootstrap_samples": len(boot_extrapolated),
    }


def scan_floquet_chaos(config: FloquetChaosConfig):
    root_rng = np.random.default_rng(config.seed)
    size_rows: list[dict[str, Any]] = []
    for modes, particles in config.sizes:
        basis = fixed_number_basis(modes, particles)
        realization_parameters: list[tuple[np.ndarray, np.ndarray]] = []
        for _ in range(config.realizations):
            disorder = root_rng.normal(size=modes)
            disorder -= disorder.mean()
            disorder *= config.disorder_strength / max(float(np.std(disorder)), 1e-15)
            phases = root_rng.normal(size=modes)
            phases -= phases.mean()
            phases *= config.phase_kick_strength / max(float(np.std(phases)), 1e-15)
            realization_parameters.append((disorder, phases))
        points = []
        for interaction in config.interactions:
            unitaries = []
            statistics = []
            eigenvectors = []
            otocs = []
            for disorder, phases in realization_parameters:
                unitary = floquet_unitary(
                    basis,
                    config.hopping,
                    interaction,
                    disorder,
                    phases,
                    config.hopping_time,
                    config.interaction_time,
                    config.boundary,
                )
                unitaries.append(unitary)
                stat = quasienergy_statistics(unitary, config.degeneracy_tolerance)
                statistics.append(stat)
                eigenvectors.append(eigenvector_diagnostics(unitary, basis))
                otocs.append(floquet_otoc(unitary, basis, config.otoc_steps))
            valid = [row for row in statistics if row["valid"]]
            realization_ratios = [row["mean_ratio"] for row in valid]
            if len(valid) != len(statistics):
                mean_ratio = None
                ci = [None, None]
            else:
                mean_ratio = float(np.mean(realization_ratios))
                boot = [
                    np.mean(
                        root_rng.choice(realization_ratios, len(realization_ratios), replace=True)
                    )
                    for _ in range(config.bootstrap_replicates)
                ]
                ci = _bootstrap_ci(boot)
            points.append(
                {
                    "interaction": float(interaction),
                    "mean_spacing_ratio": mean_ratio,
                    "spacing_ratio_ci95": ci,
                    "realization_mean_ratios": realization_ratios,
                    "degeneracy_fraction": float(
                        np.mean([row["degeneracy_fraction"] for row in statistics])
                    ),
                    "degeneracy_failure": len(valid) != len(statistics),
                    "quasienergies_by_realization": [row["phases"].tolist() for row in statistics],
                    "spacing_ratios_by_realization": [row["ratios"].tolist() for row in statistics],
                    "sff": sff_ensemble(
                        unitaries,
                        max(config.otoc_steps, 8),
                        root_rng,
                        config.bootstrap_replicates,
                        config.universality_class,
                    ),
                    "eigenvector_diagnostics": {
                        key: float(np.mean([row[key] for row in eigenvectors]))
                        for key in (
                            "mean_ipr",
                            "mean_participation_entropy",
                            "normalized_participation_entropy",
                            "mean_bipartite_entanglement_entropy",
                        )
                    },
                    "otoc_mean": np.mean(otocs, axis=0).tolist(),
                    "otoc_ci95": np.quantile(otocs, [0.025, 0.975], axis=0).T.tolist(),
                }
            )
        size_rows.append(
            {
                "label": f"M{modes}_N{particles}",
                "modes": modes,
                "particles": particles,
                "dimension": len(basis),
                "sector_rule": "fixed total particle number; spatial symmetries broken by seeded onsite disorder and kick phases",
                "points": points,
            }
        )
    if any(point["degeneracy_failure"] for size in size_rows for point in size["points"]):
        validation = "FAIL_DEGENERACY"
        for size in size_rows:
            size["chaos_scores"] = [None] * len(size["points"])
            size["entry_crossings"] = []
            size["additional_crossings"] = []
            size["entry_ci95"] = [None, None]
        edge: dict[str, Any] = {
            "status": "unresolved_degeneracy",
            "estimate": None,
            "ci95": [None, None],
            "size_estimates": [None] * len(size_rows),
            "all_crossings": {size["label"]: [] for size in size_rows},
            "reason": "at least one realization contains an unresolved near-degeneracy",
        }
    else:
        validation = "PASS"
        edge = _edge_estimate(size_rows, config, root_rng)
    if edge["estimate"] is not None:
        grid_step = float(np.min(np.diff(config.interactions)))
        ci = edge["ci95"]
        uncertainty = grid_step
        if ci[0] is not None:
            uncertainty = max(grid_step, 0.5 * (ci[1] - ci[0]))
        for size in size_rows:
            for point, score in zip(size["points"], size["chaos_scores"]):
                coordinate = (point["interaction"] - edge["estimate"]) / uncertainty
                point["d_eoc"] = float(coordinate)
                point["chaos_score"] = float(score)
                if coordinate < -0.5:
                    point["region"] = "regular"
                elif coordinate <= 0.5:
                    point["region"] = "crossover"
                elif score >= 0.75:
                    point["region"] = "chaotic"
                else:
                    point["region"] = "post-chaotic-regular"
    return {
        "model": "two-kick number-conserving Bose-Hubbard Floquet",
        "floquet_definition": "exp(-i H_U T_U) exp(-i H_J T_J)",
        "configuration": asdict(config),
        "universality_class": {
            "selected": config.universality_class,
            "reason": "real generators admit a symmetric time origin, giving antiunitary symmetry",
        },
        "sizes": size_rows,
        "edge_estimate": edge,
        "validation": validation,
        "boundary_independent_of_performance": True,
        "d_eoc_definition": "(U-U_EOC)/max(grid spacing, half CI width); negative is regular side",
    }
