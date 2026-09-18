"""Four-mode truncated-boson Floquet reservoir with a complete noisy channel."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from math import comb

import numpy as np
from scipy.linalg import expm

from .channel import apply_kraus_raw, compose_kraus, staged_channel_diagnostics
from .core import state_diagnostics
from .science import floquet_unitary


def local_basis(modes: int, cutoff: int):
    return tuple(product(range(cutoff), repeat=modes))


def _loss_kraus(cutoff: int, transmissivity: float):
    result = []
    for lost in range(cutoff):
        operator = np.zeros((cutoff, cutoff), complex)
        for occupied in range(lost, cutoff):
            operator[occupied - lost, occupied] = (
                np.sqrt(comb(occupied, lost))
                * (1 - transmissivity) ** (lost / 2)
                * transmissivity ** ((occupied - lost) / 2)
            )
        result.append(operator)
    return tuple(result)


def _tensor_stage(local, modes):
    result = []
    for choices in product(local, repeat=modes):
        operator = choices[0]
        for choice in choices[1:]:
            operator = np.kron(operator, choice)
        result.append(operator)
    return tuple(result)


@dataclass(frozen=True)
class DVFloquetConfig:
    memory_modes: int = 4
    local_cutoff: int = 3
    hopping: float = 1.0
    interaction: float = 1.0
    disorder: tuple[float, ...] = (0.11, -0.07, 0.03, -0.07)
    kick_phases: tuple[float, ...] = (0.05, -0.09, 0.12, -0.08)
    hopping_time: float = 0.37
    interaction_time: float = 0.23
    boundary: str = "open"
    collision_angle: float = 0.24
    transmissivity: float = 0.94
    evolution: str = "unconditional"
    trace_tolerance: float = 1e-9
    seed: int = 17

    def __post_init__(self):
        if self.memory_modes < 4:
            raise ValueError("principal EOC reservoir requires at least four memory modes")
        if self.local_cutoff < 3:
            raise ValueError("onsite Kerr requires local cutoff >=3")
        if len(self.disorder) != self.memory_modes or len(self.kick_phases) != self.memory_modes:
            raise ValueError("one disorder and kick phase per memory mode required")
        if self.evolution not in {"unconditional", "conditional"}:
            raise ValueError("unknown trajectory policy")
        if not 0 <= self.transmissivity <= 1:
            raise ValueError("transmissivity must lie in [0,1]")


class DVFloquetReservoir:
    """Fresh-mode collision followed by the calibrated memory Floquet and loss."""

    def __init__(self, config: DVFloquetConfig | None = None):
        self.config = config or DVFloquetConfig()
        c = self.config
        self.basis = local_basis(c.memory_modes, c.local_cutoff)
        self.joint_basis = local_basis(c.memory_modes + 1, c.local_cutoff)
        self._joint_index = {state: index for index, state in enumerate(self.joint_basis)}
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
        self.collision_unitary = self._collision_unitary()
        dimension = len(self.basis)
        self._collision_components = np.empty(
            (c.local_cutoff, c.local_cutoff, dimension, dimension), complex
        )
        for outcome in range(c.local_cutoff):
            rows = [self._joint_index[state + (outcome,)] for state in self.basis]
            for photon in range(c.local_cutoff):
                columns = [self._joint_index[state + (photon,)] for state in self.basis]
                self._collision_components[outcome, photon] = self.collision_unitary[
                    np.ix_(rows, columns)
                ]
        self.loss_kraus = _tensor_stage(
            _loss_kraus(c.local_cutoff, c.transmissivity), c.memory_modes
        )
        local_loss = _loss_kraus(c.local_cutoff, c.transmissivity)
        self.loss_stages = []
        identity = np.eye(c.local_cutoff)
        for mode in range(c.memory_modes):
            stage = []
            for local in local_loss:
                factors = [identity] * c.memory_modes
                factors[mode] = local
                operator = factors[0]
                for factor in factors[1:]:
                    operator = np.kron(operator, factor)
                stage.append(operator)
            self.loss_stages.append(tuple(stage))
        self._internal_stage = (self.internal_unitary,)
        self.reset()

    def _collision_unitary(self):
        generator = np.zeros((len(self.joint_basis), len(self.joint_basis)), complex)
        memory_mode, ancilla_mode = 0, self.config.memory_modes
        for column, state in enumerate(self.joint_basis):
            for source, destination, sign in (
                (ancilla_mode, memory_mode, 1),
                (memory_mode, ancilla_mode, -1),
            ):
                if state[source] == 0 or state[destination] + 1 >= self.config.local_cutoff:
                    continue
                changed = list(state)
                changed[source] -= 1
                changed[destination] += 1
                row = self._joint_index[tuple(changed)]
                generator[row, column] += sign * np.sqrt(state[source] * (state[destination] + 1))
        return expm(self.config.collision_angle * generator)

    def reset(self, seed: int | None = None):
        self.rng = np.random.default_rng(self.config.seed if seed is None else seed)
        self.state = np.zeros((len(self.basis), len(self.basis)), complex)
        self.state[0, 0] = 1
        self.time = 0
        return self

    def collision_kraus(self, value: float):
        if not 0 <= value <= 1:
            raise ValueError("DV input must lie in [0,1]")
        ancilla = np.zeros(self.config.local_cutoff, complex)
        ancilla[:2] = (np.sqrt(1 - value), np.sqrt(value))
        return tuple(
            np.tensordot(ancilla, components, axes=(0, 0))
            for components in self._collision_components
        )

    def complete_kraus(self, value: float):
        return compose_kraus(self.collision_kraus(value), self._internal_stage, self.loss_kraus)

    def channel_diagnostics(self, value: float):
        result = staged_channel_diagnostics(
            (self.collision_kraus(value), self._internal_stage, *self.loss_stages),
            tp_tolerance=self.config.trace_tolerance,
        )
        result["included_stages"] = [
            "fresh-ancilla injection",
            "memory-ancilla collision",
            "PNR averaging",
            "memory Floquet interaction",
            "all amplitude-damping loss sectors",
        ]
        return result

    def _features(self):
        probabilities = np.real(np.diag(self.state))
        occupations = np.asarray(self.basis, float)
        means = probabilities @ occupations
        correlations = [
            float(probabilities @ (occupations[:, left] * occupations[:, right]))
            for left in range(self.config.memory_modes)
            for right in range(left, self.config.memory_modes)
        ]
        parity = probabilities @ ((-1.0) ** occupations)
        return np.r_[means, correlations, parity, probabilities]

    def step(self, value: float):
        collision = self.collision_kraus(value)
        raw = apply_kraus_raw(self.state, collision)
        raw = self.internal_unitary @ raw @ self.internal_unitary.conj().T
        for stage in self.loss_stages:
            raw = apply_kraus_raw(raw, stage)
        raw_trace = float(np.trace(raw).real)
        if abs(raw_trace - 1) > self.config.trace_tolerance:
            raise RuntimeError(f"raw channel trace {raw_trace} violates TP tolerance")
        if self.config.evolution == "unconditional":
            self.state = raw
            outcome = None
        else:
            kraus = compose_kraus(collision, self._internal_stage, self.loss_kraus)
            branches = [operator @ self.state @ operator.conj().T for operator in kraus]
            probabilities = np.asarray([np.trace(branch).real for branch in branches])
            probabilities /= probabilities.sum()
            outcome = int(self.rng.choice(len(branches), p=probabilities))
            self.state = branches[outcome] / probabilities[outcome]
        self.time += 1
        diagnostics = state_diagnostics(self.state, self.basis, cutoff=self.config.local_cutoff)
        diagnostics["raw_output_trace"] = raw_trace
        diagnostics["local_cutoff_boundary_population"] = float(
            sum(
                probability
                for probability, state in zip(np.real(np.diag(self.state)), self.basis)
                if any(occupied == self.config.local_cutoff - 1 for occupied in state)
            )
        )
        return {
            "features": self._features(),
            "diagnostics": diagnostics,
            "outcome": outcome,
            "configuration": asdict(self.config),
            "provenance": {
                "internal_model": "same two-kick Floquet Bose-Hubbard model as closed calibration",
                "representation": "finite local truncated-boson/qudit",
                "perceval_scope": "preserved static passive control only",
            },
        }
