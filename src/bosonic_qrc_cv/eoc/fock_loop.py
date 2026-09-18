"""Non-Gaussian recurrent CV reservoir with exclusive total-photon cutoff."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import factorial
from typing import Literal

import numpy as np

from .core import (
    bose_hubbard_parts,
    coherent_amplitudes,
    density_fidelity,
    exact_unitary,
    floquet_unitary,
    reduced_density_matrix,
    state_diagnostics,
    strang_unitary,
    total_cutoff_basis,
    trace_distance,
)


@dataclass(frozen=True)
class FockLoopConfig:
    memory_modes: int = 2
    cutoff: int = 5
    hopping: float = 1.0
    interaction: float = 0.8
    disorder: tuple[float, ...] = (0.0, 0.0, 0.0)
    interval: float = 0.2
    boundary: Literal["open", "ring"] = "open"
    evolution_model: Literal["exact", "floquet", "trotter"] = "exact"
    trotter_substeps: int = 4
    resource: Literal["coherent", "squeezed", "cat", "number", "fock"] = "coherent"
    resource_strength: float = 0.35
    number_state: int = 1
    fock_amplitudes: tuple[complex, ...] = (1.0, 0.0)
    input_scale: float = 0.5
    transmissivity: float = 0.94
    seed: int = 7

    def __post_init__(self):
        if self.memory_modes < 1 or self.cutoff < 2:
            raise ValueError("memory_modes positive and cutoff>=2 required")
        if len(self.disorder) != self.memory_modes + 1:
            raise ValueError("disorder covers memory and one fresh mode")
        if not 0 <= self.transmissivity <= 1:
            raise ValueError("transmissivity must lie in [0,1]")
        if self.number_state >= self.cutoff:
            raise ValueError("number state must lie below total cutoff")


def _resource(config: FockLoopConfig, value: float) -> np.ndarray:
    cutoff = config.cutoff
    if config.resource == "coherent":
        return coherent_amplitudes(
            config.resource_strength + config.input_scale * value, cutoff
        )
    if config.resource == "squeezed":
        amplitudes = np.zeros(cutoff, complex)
        squeezing = config.resource_strength
        for photon in range(0, cutoff, 2):
            pair = photon // 2
            amplitudes[photon] = (
                np.sqrt(factorial(2 * pair))
                / (2**pair * factorial(pair) * np.sqrt(np.cosh(squeezing)))
                * (-np.tanh(squeezing)) ** pair
            )
    elif config.resource == "cat":
        plus = coherent_amplitudes(config.resource_strength, cutoff)
        minus = coherent_amplitudes(-config.resource_strength, cutoff)
        amplitudes = plus + minus
    elif config.resource == "number":
        amplitudes = np.zeros(cutoff, complex)
        amplitudes[config.number_state] = 1
    else:
        amplitudes = np.asarray(config.fock_amplitudes, complex)
        amplitudes = np.pad(amplitudes[:cutoff], (0, max(0, cutoff - len(amplitudes))))
    amplitudes *= np.exp(1j * config.input_scale * value * np.arange(cutoff))
    norm = np.linalg.norm(amplitudes)
    if norm == 0:
        raise ValueError("fresh resource has zero norm")
    return amplitudes / norm


def _loss_kraus(
    basis: tuple[tuple[int, ...], ...], transmissivity: float
) -> tuple[np.ndarray, ...]:
    from math import comb

    modes = len(basis[0])
    index = {state: i for i, state in enumerate(basis)}
    maximum = max(max(state) for state in basis)
    operators = []
    for losses in np.ndindex(*((maximum + 1,) * modes)):
        operator = np.zeros((len(basis), len(basis)), complex)
        for column, state in enumerate(basis):
            if any(loss > occupied for loss, occupied in zip(losses, state)):
                continue
            output = tuple(occupied - loss for occupied, loss in zip(state, losses))
            row = index[output]
            amplitude = 1.0
            for occupied, loss in zip(state, losses):
                amplitude *= (
                    np.sqrt(comb(occupied, loss))
                    * (1 - transmissivity) ** (loss / 2)
                    * transmissivity ** ((occupied - loss) / 2)
                )
            operator[row, column] = amplitude
        if np.any(operator):
            operators.append(operator)
    return tuple(operators)


def _ladder(basis: tuple[tuple[int, ...], ...], mode: int) -> np.ndarray:
    index = {state: i for i, state in enumerate(basis)}
    operator = np.zeros((len(basis), len(basis)), complex)
    for column, state in enumerate(basis):
        if state[mode]:
            output = list(state)
            output[mode] -= 1
            operator[index[tuple(output)], column] = np.sqrt(state[mode])
    return operator


class NonGaussianCVReservoir:
    """Persistent mixed Fock memory; the original Gaussian loop is unchanged."""

    def __init__(self, config: FockLoopConfig | None = None):
        self.config = config or FockLoopConfig()
        self.memory_basis = total_cutoff_basis(
            self.config.memory_modes, self.config.cutoff
        )
        self.joint_basis = total_cutoff_basis(
            self.config.memory_modes + 1, self.config.cutoff
        )
        parts = bose_hubbard_parts(
            self.joint_basis,
            self.config.hopping,
            self.config.interaction,
            np.asarray(self.config.disorder),
            self.config.boundary,
        )
        if self.config.evolution_model == "exact":
            self.unitary = exact_unitary(parts.total, self.config.interval)
        elif self.config.evolution_model == "floquet":
            self.unitary = floquet_unitary(parts, self.config.interval)
        else:
            self.unitary = strang_unitary(
                parts, self.config.interval, self.config.trotter_substeps
            )
        self._joint_index = {state: i for i, state in enumerate(self.joint_basis)}
        self.loss_kraus = _loss_kraus(self.memory_basis, self.config.transmissivity)
        self.reset()

    def reset(self):
        self.state = np.zeros((len(self.memory_basis), len(self.memory_basis)), complex)
        vacuum = self.memory_basis.index((0,) * self.config.memory_modes)
        self.state[vacuum, vacuum] = 1
        self.time = 0
        return self

    def _joint_density(self, resource: np.ndarray):
        joint = np.zeros((len(self.joint_basis), len(self.joint_basis)), complex)
        retained = 0.0
        for row, memory_bra in enumerate(self.memory_basis):
            for column, memory_ket in enumerate(self.memory_basis):
                memory_value = self.state[row, column]
                if memory_value == 0:
                    continue
                for ancilla_bra, bra_amplitude in enumerate(resource):
                    bra = memory_bra + (ancilla_bra,)
                    bra_index = self._joint_index.get(bra)
                    if bra_index is None:
                        continue
                    for ancilla_ket, ket_amplitude in enumerate(resource):
                        ket = memory_ket + (ancilla_ket,)
                        ket_index = self._joint_index.get(ket)
                        if ket_index is None:
                            continue
                        joint[bra_index, ket_index] += (
                            memory_value * bra_amplitude * ket_amplitude.conjugate()
                        )
        retained_raw = float(np.trace(joint).real)
        if retained_raw <= 0:
            raise RuntimeError("total cutoff removed the complete input state")
        retained = min(1.0, max(0.0, retained_raw))
        return joint / retained_raw, retained

    @staticmethod
    def _apply_channel(density, operators):
        return sum(operator @ density @ operator.conj().T for operator in operators)

    def step(self, value: float) -> dict[str, object]:
        resource = _resource(self.config, float(value))
        joint, retained = self._joint_density(resource)
        evolved = self.unitary @ joint @ self.unitary.conj().T
        reduced, basis = reduced_density_matrix(
            evolved, self.joint_basis, tuple(range(self.config.memory_modes))
        )
        if basis != self.memory_basis:
            raise RuntimeError("unexpected reduced basis ordering")
        reduced = self._apply_channel(reduced, self.loss_kraus)
        reduced /= np.trace(reduced)
        self.state = reduced
        self.time += 1
        return {
            "features": self.features(),
            "diagnostics": {
                **state_diagnostics(reduced, self.memory_basis, self.config.cutoff),
                "retained_input_norm": retained,
            },
        }

    def features(self) -> np.ndarray:
        rho, basis = self.state, self.memory_basis
        probabilities = np.real(np.diag(rho))
        occupations = np.asarray(basis)
        means = probabilities @ occupations
        number_correlations = [
            float(probabilities @ (occupations[:, i] * occupations[:, j]))
            for i in range(self.config.memory_modes)
            for j in range(i, self.config.memory_modes)
        ]
        g2 = []
        parity = []
        moments = []
        gaussian = []
        for mode in range(self.config.memory_modes):
            n = occupations[:, mode]
            numerator = float(probabilities @ (n * (n - 1)))
            g2.append(numerator / means[mode] ** 2 if means[mode] > 1e-12 else 0.0)
            parity.append(float(probabilities @ ((-1.0) ** n)))
            annihilation = _ladder(basis, mode)
            x = annihilation + annihilation.conj().T
            p = -1j * (annihilation - annihilation.conj().T)
            x_values = [
                float(np.trace(rho @ np.linalg.matrix_power(x, order)).real)
                for order in range(1, 5)
            ]
            moments.extend(x_values)
            gaussian.extend(
                [
                    x_values[0],
                    float(np.trace(rho @ p).real),
                    x_values[1] - x_values[0] ** 2,
                    float(np.trace(rho @ (p @ p)).real)
                    - float(np.trace(rho @ p).real) ** 2,
                ]
            )
        return np.r_[
            means, number_correlations, g2, parity, moments, gaussian, probabilities
        ]

    def compare_state(self, other: NonGaussianCVReservoir) -> dict[str, float]:
        union = tuple(sorted(set(self.memory_basis) | set(other.memory_basis)))
        left = align_density(self.state, self.memory_basis, union)
        right = align_density(other.state, other.memory_basis, union)
        return {
            "fidelity": density_fidelity(left, right),
            "trace_distance": trace_distance(left, right),
        }

    def provenance(self):
        return {
            "config": asdict(self.config),
            "cutoff_semantics": "exclusive total photon number",
            "hilbert_dimension": len(self.memory_basis),
            "backend": "independent SciPy Fock reference",
            "gaussian_baseline_unchanged": True,
        }


def piquasso_kerr_phases(
    cutoff: int, interaction: float, interval: float
) -> np.ndarray:
    """Execute Piquasso 8 Kerr+phase gates and return diagonal BH onsite phases."""
    import piquasso as pq

    phases = []
    simulator = pq.PureFockSimulator(d=1, config=pq.Config(cutoff=cutoff))
    for photon in range(cutoff):
        with pq.Program() as program:
            pq.Q(0) | pq.NumberState((photon,))
            pq.Q(0) | pq.Kerr(xi=-0.5 * interaction * interval)
            pq.Q(0) | pq.Phaseshifter(phi=0.5 * interaction * interval)
        state = simulator.execute(program).state
        phases.append(state.fock_amplitudes_map.get((photon,), 0.0))
    return np.asarray(phases)


def align_density(density, source_basis, destination_basis):
    destination = {state: index for index, state in enumerate(destination_basis)}
    aligned = np.zeros((len(destination_basis), len(destination_basis)), complex)
    for row, bra in enumerate(source_basis):
        for column, ket in enumerate(source_basis):
            aligned[destination[bra], destination[ket]] = density[row, column]
    return aligned
