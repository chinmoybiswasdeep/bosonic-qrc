"""Configurable fixed Perceval SLOS quantum reservoir."""
from __future__ import annotations

import importlib
from dataclasses import dataclass
from itertools import product

import numpy as np

from .config import DVConfig


def _perceval():
    try:
        return importlib.import_module("perceval")
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("perceval-quandela 1.2 is required") from exc


@dataclass(frozen=True)
class ProbabilityFeatures:
    values: np.ndarray
    outcomes: tuple[str, ...]
    sector_weights: dict[int, float]
    backend: str = "perceval.SLOS"


def pnr_outcomes(modes: int, maximum_photons: int, include_lower: bool) -> tuple[tuple[int, ...], ...]:
    """Deterministic lexicographic PNR index, including collisions."""
    states = [occupation for occupation in product(range(maximum_photons + 1), repeat=modes) if sum(occupation) <= maximum_photons]
    if not include_lower:
        states = [occupation for occupation in states if sum(occupation) == maximum_photons]
    return tuple(sorted(states, key=lambda state: (sum(state), state)))


class LinearOpticalReservoir:
    """Fixed Haar-like multimode reservoir with optional dual-rail encoding.

    The Haar unitary is decomposed by Perceval into beam splitters and phase
    shifters. SLOS supplies every exact probability; finite-shot frequencies
    come from Perceval's sampler conversion rather than local multinomial draws.
    This class is static QRP, not recurrent QRC.
    """

    def __init__(self, config: DVConfig | None = None, unitary: np.ndarray | None = None) -> None:
        config = DVConfig() if config is None else config
        self.config = config
        self.pcvl = _perceval()
        rng = np.random.default_rng(config.reservoir_seed)
        self.unitary = self._haar(config.modes, rng) if unitary is None else np.asarray(unitary)
        if not np.allclose(self.unitary.conj().T @ self.unitary, np.eye(config.modes), atol=1e-8):
            raise ValueError("reservoir matrix must be unitary")
        self.reservoir_circuit = self._decompose(self.unitary)
        self.execution_count = 0
        self.outcome_index = pnr_outcomes(config.modes, config.photons, config.include_lower_sectors)

    @staticmethod
    def _haar(modes: int, rng: np.random.Generator) -> np.ndarray:
        z = rng.normal(size=(modes, modes)) + 1j * rng.normal(size=(modes, modes))
        q, r = np.linalg.qr(z)
        diagonal = np.diag(r)
        return q * (diagonal / np.where(np.abs(diagonal) == 0, 1, np.abs(diagonal)))

    def _decompose(self, unitary: np.ndarray):
        pcvl = self.pcvl
        mzi = pcvl.BS() // (0, pcvl.PS(phi=pcvl.Parameter("phi_a"))) // pcvl.BS() // (1, pcvl.PS(phi=pcvl.Parameter("phi_b")))
        circuit = pcvl.Circuit.decomposition(unitary, mzi, phase_shifter_fn=pcvl.PS, shape="triangle")
        if circuit is None:
            raise RuntimeError("Perceval failed to decompose the reservoir unitary")
        return circuit

    def default_input(self) -> tuple[int, ...]:
        return tuple(1 if mode < self.config.photons else 0 for mode in range(self.config.modes))

    def dual_rail_input(self) -> tuple[int, ...]:
        """Return two photons in the first rail of two dual-rail pairs."""
        if self.config.modes < 4 or self.config.photons != 2:
            raise ValueError("dual-rail encoding requires at least four modes and two photons")
        state = [0] * self.config.modes
        state[0] = 1
        state[2] = 1
        return tuple(state)

    def _circuit(self, coordinates: tuple[float, float] | None):
        pcvl = self.pcvl
        circuit = pcvl.Circuit(self.config.modes, name="encoded_fixed_qrp")
        if coordinates is not None:
            if self.config.modes < 4 or self.config.photons != 2:
                raise ValueError("dual-rail coordinate encoding requires four modes and two photons")
            x, y = coordinates
            circuit.add((0, 1), pcvl.BS())
            circuit.add(0, pcvl.PS(phi=float(x)))
            circuit.add((2, 3), pcvl.BS())
            circuit.add(2, pcvl.PS(phi=float(y)))
        circuit.add(0, self.reservoir_circuit)
        return circuit

    def probabilities(self, state: tuple[int, ...] | None = None, coordinates: tuple[float, float] | None = None) -> ProbabilityFeatures:
        """Execute SLOS and return a fixed-length ordered PNR vector."""
        state = self.default_input() if state is None else state
        if len(state) != self.config.modes or sum(state) != self.config.photons:
            raise ValueError("input occupation must match configured modes and photons")
        processor = self.pcvl.Processor("SLOS", self._circuit(coordinates))
        processor.with_input(self.pcvl.BasicState(list(state)))
        sampler = self.pcvl.algorithm.Sampler(processor)
        if self.config.measurement == "exact":
            distribution = sampler.probs()["results"]
            by_tuple = {tuple(output): float(probability) for output, probability in distribution.items()}
        else:
            counts = sampler.sample_count(self.config.shots)["results"]
            by_tuple = {tuple(output): count / self.config.shots for output, count in counts.items()}
        self.execution_count += 1
        values = np.asarray([by_tuple.get(outcome, 0.0) for outcome in self.outcome_index])
        sectors = {sector: float(sum(value for outcome, value in zip(self.outcome_index, values) if sum(outcome) == sector)) for sector in range(self.config.photons + 1)}
        labels = tuple("|" + ",".join(map(str, outcome)) + ">" for outcome in self.outcome_index)
        return ProbabilityFeatures(values, labels, sectors)
