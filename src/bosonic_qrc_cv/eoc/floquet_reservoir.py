"""Four-mode total-cutoff non-Gaussian Floquet reservoir."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from math import comb

import numpy as np
from scipy.linalg import expm

from .channel import apply_kraus_raw, compose_kraus, staged_channel_diagnostics
from .core import state_diagnostics, total_cutoff_basis
from .science import floquet_unitary


def _loss_kraus(basis, transmissivity):
    index = {state: position for position, state in enumerate(basis)}
    modes = len(basis[0])
    maximum = max(max(state) for state in basis)
    result = []
    for losses in product(range(maximum + 1), repeat=modes):
        operator = np.zeros((len(basis), len(basis)), complex)
        for column, state in enumerate(basis):
            if any(lost > occupied for lost, occupied in zip(losses, state)):
                continue
            output = tuple(occupied - lost for occupied, lost in zip(state, losses))
            amplitude = 1.0
            for occupied, lost in zip(state, losses):
                amplitude *= (
                    np.sqrt(comb(occupied, lost))
                    * (1 - transmissivity) ** (lost / 2)
                    * transmissivity ** ((occupied - lost) / 2)
                )
            operator[index[output], column] = amplitude
        if np.any(operator):
            result.append(operator)
    return tuple(result)


def _annihilation(basis, mode):
    index = {state: position for position, state in enumerate(basis)}
    operator = np.zeros((len(basis), len(basis)), complex)
    for column, state in enumerate(basis):
        if state[mode] == 0:
            continue
        changed = list(state)
        changed[mode] -= 1
        operator[index[tuple(changed)], column] = np.sqrt(state[mode])
    return operator


def _mode_loss_stages(basis, transmissivity):
    index = {state: position for position, state in enumerate(basis)}
    maximum = max(max(state) for state in basis)
    stages = []
    for mode in range(len(basis[0])):
        stage = []
        for lost in range(maximum + 1):
            operator = np.zeros((len(basis), len(basis)), complex)
            for column, state in enumerate(basis):
                if lost > state[mode]:
                    continue
                changed = list(state)
                changed[mode] -= lost
                operator[index[tuple(changed)], column] = (
                    np.sqrt(comb(state[mode], lost))
                    * (1 - transmissivity) ** (lost / 2)
                    * transmissivity ** ((state[mode] - lost) / 2)
                )
            if np.any(operator):
                stage.append(operator)
        stages.append(tuple(stage))
    return tuple(stages)


@dataclass(frozen=True)
class CVFloquetConfig:
    memory_modes: int = 4
    cutoff: int = 4
    hopping: float = 1.0
    interaction: float = 1.0
    disorder: tuple[float, ...] = (0.11, -0.07, 0.03, -0.07)
    kick_phases: tuple[float, ...] = (0.05, -0.09, 0.12, -0.08)
    hopping_time: float = 0.37
    interaction_time: float = 0.23
    boundary: str = "open"
    collision_angle: float = 0.22
    resource_strength: float = 0.12
    input_scale: float = 0.10
    transmissivity: float = 0.94
    trace_tolerance: float = 1e-9
    seed: int = 17

    def __post_init__(self):
        if self.memory_modes < 4:
            raise ValueError(
                "principal EOC reservoir requires at least four memory modes"
            )
        if self.cutoff < 3:
            raise ValueError("total cutoff must retain Kerr-active occupations")
        if (
            len(self.disorder) != self.memory_modes
            or len(self.kick_phases) != self.memory_modes
        ):
            raise ValueError("one disorder and kick phase per memory mode required")
        if not 0 <= self.transmissivity <= 1:
            raise ValueError("transmissivity must lie in [0,1]")


class CVFloquetReservoir:
    """Projected-unitary fresh-mode encoding, collision, memory Floquet, and loss."""

    def __init__(self, config: CVFloquetConfig | None = None):
        self.config = config or CVFloquetConfig()
        c = self.config
        self.basis = total_cutoff_basis(c.memory_modes, c.cutoff)
        self.joint_basis = total_cutoff_basis(c.memory_modes + 1, c.cutoff)
        self._joint_index = {
            state: index for index, state in enumerate(self.joint_basis)
        }
        self._memory_index = {state: index for index, state in enumerate(self.basis)}
        self.internal_unitary = floquet_unitary(
            self.basis,
            c.hopping,
            c.interaction,
            np.asarray(c.disorder),
            np.asarray(c.kick_phases),
            c.hopping_time,
            c.interaction_time,
            c.boundary,
        )
        self.loss_kraus = _loss_kraus(self.basis, c.transmissivity)
        self.loss_stages = _mode_loss_stages(self.basis, c.transmissivity)
        self._internal_stage = (self.internal_unitary,)
        self._joint_annihilation = _annihilation(self.joint_basis, c.memory_modes)
        memory_annihilation = _annihilation(self.joint_basis, 0)
        transfer = memory_annihilation.conj().T @ self._joint_annihilation
        # Projection to a total-Fock subspace spoils exact commutation of
        # different-mode ladder matrices at the cutoff boundary.  Construct
        # A - A^dagger explicitly so the collision remains exactly unitary.
        generator = transfer - transfer.conj().T
        self.collision_unitary = expm(c.collision_angle * generator)
        preparation_hamiltonian = 1j * (
            self._joint_annihilation.conj().T - self._joint_annihilation
        )
        prep_values, prep_vectors = np.linalg.eigh(preparation_hamiltonian)
        embedded_columns = [self._joint_index[state + (0,)] for state in self.basis]
        embedding = np.eye(len(self.joint_basis), dtype=complex)[:, embedded_columns]
        self._prep_values = prep_values
        self._collision_prep_left = self.collision_unitary @ prep_vectors
        self._collision_prep_right = prep_vectors.conj().T @ embedding
        self._outcome_rows = tuple(
            np.asarray(
                [self._joint_index.get(state + (outcome,), -1) for state in self.basis]
            )
            for outcome in range(c.cutoff)
        )
        self.memory_ladders = tuple(
            _annihilation(self.basis, mode) for mode in range(c.memory_modes)
        )
        self.reset()

    def reset(self, seed: int | None = None):
        self.rng = np.random.default_rng(self.config.seed if seed is None else seed)
        self.state = np.zeros((len(self.basis), len(self.basis)), complex)
        self.state[0, 0] = 1
        self.time = 0
        return self

    def collision_kraus(self, value: float):
        if not -1 <= value <= 1:
            raise ValueError("CV input must lie in [-1,1]")
        alpha = self.config.resource_strength + self.config.input_scale * value
        phase = np.exp(-1j * alpha * self._prep_values)
        joint_columns = self._collision_prep_left @ (
            phase[:, None] * self._collision_prep_right
        )
        operators = []
        for rows in self._outcome_rows:
            operator = np.zeros((len(self.basis), len(self.basis)), complex)
            valid = rows >= 0
            operator[valid] = joint_columns[rows[valid]]
            operators.append(operator)
        return tuple(operators)

    def complete_kraus(self, value: float):
        return compose_kraus(
            self.collision_kraus(value), self._internal_stage, self.loss_kraus
        )

    def channel_diagnostics(self, value: float):
        result = staged_channel_diagnostics(
            (self.collision_kraus(value), self._internal_stage, *self.loss_stages),
            tp_tolerance=self.config.trace_tolerance,
        )
        result["included_stages"] = [
            "projected-unitary fresh coherent encoding",
            "memory-ancilla beam splitter",
            "unconditional PNR reduction",
            "memory Floquet Bose-Hubbard/Kerr",
            "all total-cutoff loss sectors",
        ]
        return result

    def _features(self):
        probabilities = np.real(np.diag(self.state))
        occupations = np.asarray(self.basis, float)
        means = probabilities @ occupations
        correlations = []
        g2 = []
        for left in range(self.config.memory_modes):
            for right in range(left, self.config.memory_modes):
                value = float(
                    probabilities @ (occupations[:, left] * occupations[:, right])
                )
                correlations.append(value)
                denominator = means[left] * means[right]
                g2.append(value / denominator if denominator > 1e-12 else 0.0)
        parity = probabilities @ ((-1.0) ** occupations)
        quadrature = []
        for annihilation in self.memory_ladders:
            q = annihilation + annihilation.conj().T
            quadrature.extend(
                [
                    float(np.trace(self.state @ q).real),
                    float(np.trace(self.state @ np.linalg.matrix_power(q, 2)).real),
                    float(np.trace(self.state @ np.linalg.matrix_power(q, 3)).real),
                    float(np.trace(self.state @ np.linalg.matrix_power(q, 4)).real),
                ]
            )
        return np.r_[means, correlations, g2, parity, quadrature, probabilities]

    def step(self, value: float):
        collision = self.collision_kraus(value)
        raw = apply_kraus_raw(self.state, collision)
        raw = self.internal_unitary @ raw @ self.internal_unitary.conj().T
        for stage in self.loss_stages:
            raw = apply_kraus_raw(raw, stage)
        raw_trace = float(np.trace(raw).real)
        if abs(raw_trace - 1) > self.config.trace_tolerance:
            raise RuntimeError(f"raw channel trace {raw_trace} violates TP tolerance")
        self.state = raw
        self.time += 1
        diagnostics = state_diagnostics(
            self.state, self.basis, cutoff=self.config.cutoff
        )
        diagnostics["raw_output_trace"] = raw_trace
        return {
            "features": self._features(),
            "diagnostics": diagnostics,
            "configuration": asdict(self.config),
            "provenance": {
                "internal_model": "same two-kick Floquet Bose-Hubbard model as closed calibration",
                "backend": "SciPy total-Fock adapter; Piquasso 8.0.1 Kerr convention independently tested",
                "non_gaussian_mechanism": "deterministic onsite n(n-1) Kerr kick",
                "postselection": False,
            },
        }

    def non_gaussian_witness(self, value: float = 0.3):
        interacting = self.step(value)["features"]
        reference = CVFloquetReservoir(
            CVFloquetConfig(**{**asdict(self.config), "interaction": 0.0})
        ).step(value)["features"]
        return {
            "feature_l2_change_from_u0": float(np.linalg.norm(interacting - reference)),
            "changed": bool(np.linalg.norm(interacting - reference) > 1e-10),
            "witness_scope": "deterministic feature change; not itself a chaos diagnostic",
        }
