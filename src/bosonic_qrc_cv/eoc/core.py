"""Exact small-system bosonic many-body reference implementation.

The closed core uses fixed-total-particle sectors and therefore has no Fock
cutoff error. Variable-number helpers use an exclusive total-photon cutoff.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import factorial

import numpy as np
from scipy.linalg import expm
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import expm_multiply


def fixed_number_basis(modes: int, particles: int) -> tuple[tuple[int, ...], ...]:
    if modes < 1 or particles < 0:
        raise ValueError("modes must be positive and particles nonnegative")
    return tuple(
        state
        for state in product(range(particles + 1), repeat=modes)
        if sum(state) == particles
    )


def total_cutoff_basis(modes: int, cutoff: int) -> tuple[tuple[int, ...], ...]:
    """Occupation basis with exclusive total photon number sum(n)<cutoff."""
    if modes < 1 or cutoff < 1:
        raise ValueError("modes and cutoff must be positive")
    return tuple(
        state for state in product(range(cutoff), repeat=modes) if sum(state) < cutoff
    )


def _edges(modes: int, boundary: str) -> tuple[tuple[int, int], ...]:
    if boundary not in {"open", "ring"}:
        raise ValueError("boundary must be open or ring")
    edges = {(site, site + 1) for site in range(modes - 1)}
    if boundary == "ring" and modes > 2:
        edges.add((0, modes - 1))
    return tuple(sorted(edges))


@dataclass(frozen=True)
class HamiltonianParts:
    interaction: np.ndarray
    linear: np.ndarray

    @property
    def total(self) -> np.ndarray:
        return self.interaction + self.linear


def bose_hubbard_parts(
    basis: tuple[tuple[int, ...], ...],
    hopping: float,
    interaction: float,
    disorder: np.ndarray | None = None,
    boundary: str = "open",
) -> HamiltonianParts:
    """Build H_int and H_hop+H_dis in an explicitly supplied occupation basis."""
    if not basis:
        raise ValueError("basis must not be empty")
    modes = len(basis[0])
    if any(len(state) != modes for state in basis):
        raise ValueError("inconsistent basis")
    epsilon = np.zeros(modes) if disorder is None else np.asarray(disorder, float)
    if epsilon.shape != (modes,) or not np.all(np.isfinite(epsilon)):
        raise ValueError("disorder must contain one finite value per mode")
    index = {state: position for position, state in enumerate(basis)}
    diagonal = np.zeros(len(basis))
    linear = np.zeros((len(basis), len(basis)), dtype=complex)
    for column, state in enumerate(basis):
        occupation = np.asarray(state)
        diagonal[column] = 0.5 * interaction * np.sum(occupation * (occupation - 1))
        linear[column, column] = float(epsilon @ occupation)
        for left, right in _edges(modes, boundary):
            for source, destination in ((right, left), (left, right)):
                if state[source] == 0:
                    continue
                changed = list(state)
                changed[source] -= 1
                changed[destination] += 1
                row = index.get(tuple(changed))
                if row is not None:
                    amplitude = np.sqrt(state[source] * (state[destination] + 1))
                    linear[row, column] += -hopping * amplitude
    interaction_matrix = np.diag(diagonal.astype(complex))
    if not np.allclose(linear, linear.conj().T):
        raise RuntimeError("constructed hopping Hamiltonian is not Hermitian")
    return HamiltonianParts(interaction_matrix, linear)


def bose_hubbard_hamiltonian(*args, **kwargs) -> np.ndarray:
    return bose_hubbard_parts(*args, **kwargs).total


def exact_unitary(hamiltonian: np.ndarray, interval: float) -> np.ndarray:
    return expm(-1j * np.asarray(hamiltonian) * interval)


def exact_action(
    hamiltonian: np.ndarray, state: np.ndarray, interval: float
) -> np.ndarray:
    return expm_multiply(-1j * csr_matrix(hamiltonian) * interval, np.asarray(state))


def floquet_unitary(parts: HamiltonianParts, interval: float) -> np.ndarray:
    half = expm(-0.5j * interval * parts.interaction)
    return half @ expm(-1j * interval * parts.linear) @ half


def strang_unitary(
    parts: HamiltonianParts, interval: float, substeps: int
) -> np.ndarray:
    if substeps < 1:
        raise ValueError("substeps must be positive")
    step = interval / substeps
    single = floquet_unitary(parts, step)
    return np.linalg.matrix_power(single, substeps)


def unitarity_error(unitary: np.ndarray) -> float:
    identity = np.eye(len(unitary))
    return float(np.linalg.norm(unitary.conj().T @ unitary - identity))


def reflection_blocks(
    matrix: np.ndarray, basis: tuple[tuple[int, ...], ...], tolerance: float = 1e-10
) -> dict[str, np.ndarray]:
    """Resolve open-chain reflection parity without mixing sectors."""
    index = {state: position for position, state in enumerate(basis)}
    reflection = np.zeros_like(matrix, dtype=complex)
    for column, state in enumerate(basis):
        reflection[index[state[::-1]], column] = 1
    if np.linalg.norm(matrix @ reflection - reflection @ matrix) > tolerance:
        raise ValueError("Hamiltonian does not possess reflection symmetry")
    plus, minus, visited = [], [], set()
    for state in basis:
        if state in visited:
            continue
        partner = state[::-1]
        visited.update((state, partner))
        first = np.zeros(len(basis), complex)
        first[index[state]] = 1
        if partner == state:
            plus.append(first)
        else:
            second = np.zeros(len(basis), complex)
            second[index[partner]] = 1
            plus.append((first + second) / np.sqrt(2))
            minus.append((first - second) / np.sqrt(2))
    blocks = {"reflection_even": np.asarray(plus).conj() @ matrix @ np.asarray(plus).T}
    if minus:
        blocks["reflection_odd"] = (
            np.asarray(minus).conj() @ matrix @ np.asarray(minus).T
        )
    return blocks


def spacing_ratios(eigenvalues: np.ndarray, edge_fraction: float = 0.1) -> np.ndarray:
    values = np.sort(np.asarray(eigenvalues, float))
    if not 0 <= edge_fraction < 0.5:
        raise ValueError("edge_fraction must be in [0,0.5)")
    trim = int(np.floor(edge_fraction * len(values)))
    values = values[trim : len(values) - trim if trim else None]
    spacings = np.diff(values)
    spacings = spacings[spacings > 1e-12]
    if len(spacings) < 2:
        return np.asarray([], float)
    return np.minimum(spacings[:-1], spacings[1:]) / np.maximum(
        spacings[:-1], spacings[1:]
    )


def symmetry_resolved_level_statistics(
    blocks: dict[str, np.ndarray], edge_fraction: float = 0.1
) -> dict[str, object]:
    ratios = {
        name: spacing_ratios(np.linalg.eigvalsh(block), edge_fraction)
        for name, block in blocks.items()
    }
    pooled = np.concatenate([value for value in ratios.values() if len(value)])
    return {
        "by_sector": {key: value.tolist() for key, value in ratios.items()},
        "mean_ratio": float(np.mean(pooled)) if len(pooled) else None,
        "ratios": pooled.tolist(),
        "edge_fraction": edge_fraction,
    }


def circular_phase_ratios(unitary: np.ndarray) -> np.ndarray:
    phases = np.sort(np.mod(np.angle(np.linalg.eigvals(unitary)), 2 * np.pi))
    spacings = np.diff(np.r_[phases, phases[0] + 2 * np.pi])
    return np.minimum(spacings, np.roll(spacings, -1)) / np.maximum(
        spacings, np.roll(spacings, -1)
    )


def spectral_form_factor(
    spectra: list[np.ndarray], times: np.ndarray
) -> dict[str, np.ndarray]:
    """Ensemble averaged unconnected and connected spectral form factors."""
    times = np.asarray(times, float)
    traces = np.asarray(
        [
            [np.sum(np.exp(-1j * time * spectrum)) for time in times]
            for spectrum in spectra
        ]
    )
    unconnected = np.mean(np.abs(traces) ** 2, axis=0)
    connected = unconnected - np.abs(np.mean(traces, axis=0)) ** 2
    return {"times": times, "unconnected": unconnected, "connected": connected}


def number_operator(basis: tuple[tuple[int, ...], ...], mode: int) -> np.ndarray:
    return np.diag([state[mode] for state in basis]).astype(complex)


def parity_operator(basis: tuple[tuple[int, ...], ...], mode: int) -> np.ndarray:
    return np.diag([(-1) ** state[mode] for state in basis]).astype(complex)


def infinite_temperature_otoc(
    unitary: np.ndarray, observable_w: np.ndarray, observable_v: np.ndarray
) -> float:
    evolved = unitary.conj().T @ observable_w @ unitary
    commutator = evolved @ observable_v - observable_v @ evolved
    return float(np.real(-np.trace(commutator @ commutator) / len(unitary)))


def reduced_density_matrix(
    density: np.ndarray,
    basis: tuple[tuple[int, ...], ...],
    keep: tuple[int, ...],
) -> tuple[np.ndarray, tuple[tuple[int, ...], ...]]:
    """Partial trace in an occupation basis, including total-cutoff bases."""
    keep = tuple(keep)
    discard = tuple(mode for mode in range(len(basis[0])) if mode not in keep)
    reduced_basis = tuple(sorted({tuple(state[i] for i in keep) for state in basis}))
    reduced_index = {state: i for i, state in enumerate(reduced_basis)}
    result = np.zeros((len(reduced_basis), len(reduced_basis)), complex)
    for row, bra in enumerate(basis):
        bra_discard = tuple(bra[i] for i in discard)
        for column, ket in enumerate(basis):
            if bra_discard != tuple(ket[i] for i in discard):
                continue
            bra_keep = tuple(bra[i] for i in keep)
            ket_keep = tuple(ket[i] for i in keep)
            result[reduced_index[bra_keep], reduced_index[ket_keep]] += density[
                row, column
            ]
    return result, reduced_basis


def state_diagnostics(
    density: np.ndarray,
    basis: tuple[tuple[int, ...], ...],
    cutoff: int | None = None,
) -> dict[str, float]:
    density = np.asarray(density)
    probabilities = np.real(np.diag(density))
    totals = np.asarray([sum(state) for state in basis])
    trace = float(np.trace(density).real)
    eigenvalues = np.linalg.eigvalsh((density + density.conj().T) / 2)
    shell = 0.0
    if cutoff is not None:
        shell = float(probabilities[totals == cutoff - 1].sum())
    return {
        "trace": trace,
        "hermiticity_error": float(np.linalg.norm(density - density.conj().T)),
        "minimum_eigenvalue": float(eigenvalues.min()),
        "boundary_shell_population": shell,
        "mean_photon_number": float(probabilities @ totals),
        "maximum_supported_photon_number": float(totals.max()),
    }


def trace_distance(left: np.ndarray, right: np.ndarray) -> float:
    return float(0.5 * np.linalg.svd(left - right, compute_uv=False).sum())


def density_fidelity(left: np.ndarray, right: np.ndarray) -> float:
    values, vectors = np.linalg.eigh((left + left.conj().T) / 2)
    root = (vectors * np.sqrt(np.clip(values, 0, None))) @ vectors.conj().T
    middle = (root @ right @ root + root @ right.conj().T @ root) / 2
    middle_values = np.linalg.eigvalsh(middle)
    return float(np.sum(np.sqrt(np.clip(middle_values, 0, None))) ** 2)


def kraus_completeness(kraus: tuple[np.ndarray, ...]) -> float:
    total = sum(operator.conj().T @ operator for operator in kraus)
    return float(np.linalg.norm(total - np.eye(total.shape[0])))


def channel_superoperator(kraus: tuple[np.ndarray, ...]) -> np.ndarray:
    return sum(np.kron(operator.conj(), operator) for operator in kraus)


def channel_spectrum(kraus: tuple[np.ndarray, ...]) -> dict[str, float]:
    superoperator = channel_superoperator(kraus)
    eigenvalues = np.sort(np.abs(np.linalg.eigvals(superoperator)))[::-1]
    singular_values = np.linalg.svd(superoperator, compute_uv=False)
    return {
        "subleading_eigenvalue_modulus": float(eigenvalues[1]),
        "eigenvalue_mixing_gap": float(1 - eigenvalues[1]),
        "subleading_singular_value": float(singular_values[1]),
        "singular_value_gap": float(1 - singular_values[1]),
    }


def coherent_amplitudes(alpha: complex, cutoff: int) -> np.ndarray:
    values = np.asarray(
        [
            np.exp(-(abs(alpha) ** 2) / 2) * alpha**n / np.sqrt(factorial(n))
            for n in range(cutoff)
        ],
        complex,
    )
    return values / np.linalg.norm(values)
