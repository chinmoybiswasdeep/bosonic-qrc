"""Piquasso execution adapter for the CV feedback-loop reservoir.

No state evolution occurs in this module outside Piquasso.  We only compose a
Gaussian preparation from the backend's reduced previous state, then execute
the input coupling, fixed interferometer, active gates and loss using
``GaussianSimulator.execute``.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass

import numpy as np

from .config import CVConfig


def _piquasso():
    """Load Piquasso lazily so non-backend tooling remains importable."""
    try:
        return importlib.import_module("piquasso")
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("Piquasso 8.0.1 is required for CV execution") from exc


@dataclass
class FeatureRecord:
    """One backend-derived observation and minimal provenance."""

    feature: np.ndarray
    covariance: np.ndarray
    backend: str


class GaussianLoopReservoir:
    """Fixed Gaussian reservoir with a persistent state and fresh input mode.

    The persistent ``N`` modes are coupled to a fresh squeezed/displaced mode.
    After a Piquasso execution the fresh output is traced out using
    ``GaussianState.reduced``; consequently later features depend causally on
    prior inputs.  The readout is deliberately absent from this class.
    """

    def __init__(self, config: CVConfig) -> None:
        self.config = config
        self._pq = _piquasso()
        self._rng = np.random.default_rng(config.seed)
        self._unitary = self._fixed_unitary(config.modes)
        self._memory_mean: np.ndarray | None = None
        self._memory_covariance: np.ndarray | None = None
        self.execution_count = 0

    def _fixed_unitary(self, d: int) -> np.ndarray:
        matrix = self._rng.normal(size=(d, d)) + 1j * self._rng.normal(size=(d, d))
        q, r = np.linalg.qr(matrix)
        return q * np.diag(r) / np.abs(np.diag(r))

    def reset(self) -> None:
        """Forget the retained state before a new independent sequence."""
        self._memory_mean = None
        self._memory_covariance = None

    def _input_gate(self, value: float):
        pq = self._pq
        if self.config.encoding == "angle":
            return pq.Squeezing(r=self.config.input_scale, phi=3 * np.pi * value / 4)
        if self.config.encoding == "amplitude":
            return pq.Squeezing(r=max(0.0, self.config.input_scale * (value + 1) / 2))
        return pq.Displacement(r=self.config.input_scale * value, phi=0.0)

    def step(self, value: float) -> FeatureRecord:
        """Execute one physical timestep and return upper-triangular x covariance.

        This is an expectation-feature mode: covariance is obtained from the
        Piquasso result state, not an independently implemented recurrence.
        """
        pq = self._pq
        d = self.config.modes
        with pq.Program() as program:
            if self._memory_mean is None:
                pq.Q() | pq.Vacuum()
            else:
                mean = np.concatenate([self._memory_mean, np.zeros(2)])
                covariance = np.eye(2 * (d + 1))
                covariance[: 2 * d, : 2 * d] = self._memory_covariance
                pq.Q() | pq.Mean(mean)
                pq.Q() | pq.Covariance(covariance)
            pq.Q(d) | self._input_gate(float(value))
            pq.Q(0, d) | pq.Beamsplitter(theta=np.arcsin(np.sqrt(self.config.reflectivity)))
            pq.Q(*range(d)) | pq.Interferometer(self._unitary)
            for mode in range(d):
                pq.Q(mode) | pq.Squeezing(r=self.config.active_squeezing)
            if self.config.loss:
                for mode in range(d):
                    pq.Q(mode) | pq.Attenuator(theta=np.arccos(np.sqrt(1 - self.config.loss)))

        result = pq.GaussianSimulator(d=d + 1).execute(program)
        self.execution_count += 1
        retained = result.state.reduced(tuple(range(d)))
        self._memory_mean = retained.xpxp_mean_vector.copy()
        self._memory_covariance = retained.xpxp_covariance_matrix.copy()
        x_cov = self._memory_covariance[0::2, 0::2]
        indices = np.triu_indices(d)
        return FeatureRecord(x_cov[indices], x_cov, "piquasso.GaussianSimulator")

    def transform(self, values: np.ndarray, washout: int = 0) -> np.ndarray:
        """Run a sequence through the backend, discarding washout observations."""
        self.reset()
        records = [self.step(float(value)).feature for value in values]
        return np.asarray(records[washout:])
