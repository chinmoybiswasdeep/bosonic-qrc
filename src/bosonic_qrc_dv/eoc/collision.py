"""Exact recurrent DV collision model with persistent mixed-state memory."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from math import comb

import numpy as np

from .core import (
    bose_hubbard_parts,
    channel_spectrum,
    exact_unitary,
    floquet_unitary,
    kraus_completeness,
    state_diagnostics,
    strang_unitary,
)


@dataclass(frozen=True)
class CollisionConfig:
    memory_modes: int = 2
    local_cutoff: int = 3
    hopping: float = 1.0
    interaction: float = 1.0
    disorder: tuple[float, ...] = (0.0, 0.0, 0.0)
    interval: float = 0.25
    boundary: str = "open"
    evolution_model: str = "exact"
    trotter_substeps: int = 4
    memory_transmissivity: float = 0.92
    evolution: str = "unconditional"
    seed: int = 13

    def __post_init__(self):
        if self.memory_modes < 1 or self.local_cutoff < 2:
            raise ValueError("positive memory modes and local_cutoff>=2 required")
        if len(self.disorder) != self.memory_modes + 1:
            raise ValueError("disorder includes memory plus one fresh ancilla")
        if self.boundary not in {"open", "ring"}:
            raise ValueError("boundary must be open or ring")
        if self.evolution_model not in {"exact", "floquet", "trotter"}:
            raise ValueError("unknown evolution model")
        if self.evolution not in {"unconditional", "conditional"}:
            raise ValueError("unknown trajectory policy")
        if not 0 <= self.memory_transmissivity <= 1:
            raise ValueError("memory_transmissivity must be in [0,1]")


def local_basis(modes: int, cutoff: int) -> tuple[tuple[int, ...], ...]:
    return tuple(product(range(cutoff), repeat=modes))


def amplitude_damping_kraus(cutoff: int, transmissivity: float):
    operators = []
    for lost in range(cutoff):
        operator = np.zeros((cutoff, cutoff), complex)
        for occupied in range(lost, cutoff):
            operator[occupied - lost, occupied] = (
                np.sqrt(comb(occupied, lost))
                * (1 - transmissivity) ** (lost / 2)
                * transmissivity ** ((occupied - lost) / 2)
            )
        operators.append(operator)
    return tuple(operators)


class RecurrentDVReservoir:
    """Persistent memory with a fresh input mode and exact Kraus reduction.

    The recurrent Fock layer is a documented SciPy dense reference because
    Perceval 1.2 SLOS does not expose persistent mixed-state Kerr reduction.
    Existing Perceval static QRP remains the passive-optics control.
    """

    def __init__(self, config: CollisionConfig | None = None):
        self.config = config or CollisionConfig()
        self.memory_basis = local_basis(self.config.memory_modes, self.config.local_cutoff)
        self.joint_basis = local_basis(self.config.memory_modes + 1, self.config.local_cutoff)
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
            self.unitary = strang_unitary(parts, self.config.interval, self.config.trotter_substeps)
        self._joint_index = {state: i for i, state in enumerate(self.joint_basis)}
        self.loss_kraus = self._memory_loss_kraus()
        self.reset()

    def reset(self, seed: int | None = None):
        self.rng = np.random.default_rng(self.config.seed if seed is None else seed)
        self.state = np.zeros((len(self.memory_basis), len(self.memory_basis)), complex)
        vacuum = self.memory_basis.index((0,) * self.config.memory_modes)
        self.state[vacuum, vacuum] = 1
        self.time = 0
        return self

    def _memory_loss_kraus(self):
        local = amplitude_damping_kraus(self.config.local_cutoff, self.config.memory_transmissivity)
        result = []
        for choices in product(local, repeat=self.config.memory_modes):
            operator = choices[0]
            for choice in choices[1:]:
                operator = np.kron(operator, choice)
            result.append(operator)
        return tuple(result)

    def collision_kraus(self, value: float) -> tuple[np.ndarray, ...]:
        if not 0 <= value <= 1:
            raise ValueError("DV collision input must be in [0,1]")
        ancilla = np.zeros(self.config.local_cutoff, complex)
        ancilla[0], ancilla[1] = np.sqrt(1 - value), np.sqrt(value)
        operators = []
        for outcome in range(self.config.local_cutoff):
            operator = np.zeros((len(self.memory_basis), len(self.memory_basis)), complex)
            for row, memory_out in enumerate(self.memory_basis):
                joint_out = self._joint_index[memory_out + (outcome,)]
                for column, memory_in in enumerate(self.memory_basis):
                    for occupied, amplitude in enumerate(ancilla):
                        joint_in = self._joint_index[memory_in + (occupied,)]
                        operator[row, column] += self.unitary[joint_out, joint_in] * amplitude
            operators.append(operator)
        return tuple(operators)

    @staticmethod
    def _apply_channel(density, operators):
        return sum(operator @ density @ operator.conj().T for operator in operators)

    def step(self, value: float, outcome: int | None = None) -> dict[str, object]:
        operators = self.collision_kraus(float(value))
        completeness = kraus_completeness(operators)
        branch_probabilities = np.asarray(
            [np.trace(operator @ self.state @ operator.conj().T).real for operator in operators]
        )
        if self.config.evolution == "unconditional":
            updated = self._apply_channel(self.state, operators)
            selected = None
        else:
            if outcome is None:
                selected = int(self.rng.choice(len(operators), p=branch_probabilities))
            else:
                selected = int(outcome)
            probability = branch_probabilities[selected]
            if probability <= 0:
                raise ValueError("selected PNR branch has zero probability")
            operator = operators[selected]
            updated = operator @ self.state @ operator.conj().T / probability
        updated = self._apply_channel(updated, self.loss_kraus)
        updated /= np.trace(updated)
        self.state = updated
        self.time += 1
        probabilities = np.real(np.diag(updated))
        occupations = np.asarray(self.memory_basis)
        means = probabilities @ occupations
        correlations = [
            float(probabilities @ (occupations[:, i] * occupations[:, j]))
            for i in range(self.config.memory_modes)
            for j in range(i, self.config.memory_modes)
        ]
        parity = [
            float(probabilities @ ((-1.0) ** occupations[:, mode]))
            for mode in range(self.config.memory_modes)
        ]
        features = np.r_[means, correlations, parity, probabilities]
        diagnostics = state_diagnostics(updated, self.memory_basis)
        diagnostics["local_cutoff_boundary_population"] = float(
            probabilities[np.any(occupations == self.config.local_cutoff - 1, axis=1)].sum()
        )
        return {
            "features": features,
            "pnr_probabilities": probabilities,
            "outcome": selected,
            "outcome_probabilities": branch_probabilities,
            "kraus_completeness_error": completeness,
            "diagnostics": diagnostics,
        }

    def channel_diagnostics(self, value: float) -> dict[str, float]:
        return channel_spectrum(self.collision_kraus(value))

    def provenance(self) -> dict[str, object]:
        return {
            "config": asdict(self.config),
            "backend_partition": {
                "recurrent_interacting_layer": "SciPy dense Fock reference",
                "static_passive_control": "Perceval SLOS",
            },
            "cutoff_semantics": "exclusive local occupation per mode",
            "hilbert_dimension": len(self.memory_basis),
        }
