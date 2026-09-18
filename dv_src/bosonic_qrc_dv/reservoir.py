"""Perceval-only output-probability feature extraction."""
from __future__ import annotations

import importlib
from dataclasses import dataclass

import numpy as np


def _perceval():
    try:
        return importlib.import_module("perceval")
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("perceval-quandela is required for DV execution") from exc


@dataclass(frozen=True)
class ProbabilityFeatures:
    values: np.ndarray
    outcomes: tuple[str, ...]
    backend: str = "perceval.SLOS"


class LinearOpticalReservoir:
    """Fixed two-mode interferometer evaluated exclusively by Perceval SLOS.

    It is a *static QRP/QELM*, not recurrent QRC: each call prepares a Fock
    input, executes the fixed circuit through ``Sampler.probs()``, and exposes
    PNR probabilities as a ridge-compatible classical feature vector.
    """

    def __init__(self, theta: float = np.pi / 2, phase: float = 0.0) -> None:
        self.pcvl = _perceval()
        self.theta = float(theta)
        self.phase = float(phase)
        self.execution_count = 0
        self.circuit = self._build_circuit()

    def _build_circuit(self):
        pcvl = self.pcvl
        circuit = pcvl.Circuit(2, name="fixed_dv_reservoir")
        circuit.add((0, 1), pcvl.BS(theta=self.theta))
        circuit.add(0, pcvl.PS(phi=self.phase))
        return circuit

    def probabilities(self, state: tuple[int, int] = (1, 1)) -> ProbabilityFeatures:
        """Run an input Fock state on SLOS and return ordered PNR probabilities."""
        pcvl = self.pcvl
        processor = pcvl.Processor("SLOS", self.circuit)
        processor.with_input(pcvl.BasicState(list(state)))
        result = pcvl.algorithm.Sampler(processor).probs()["results"]
        self.execution_count += 1
        ordered = sorted(((str(outcome), float(probability)) for outcome, probability in result.items()))
        return ProbabilityFeatures(np.asarray([p for _, p in ordered]), tuple(k for k, _ in ordered))
