"""Piquasso-native N+N mode Gaussian feedback reservoir."""
from __future__ import annotations

import importlib
from dataclasses import dataclass

import numpy as np

from .config import CVConfig


def _piquasso():
    try:
        return importlib.import_module("piquasso")
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Piquasso 8.0.1 is required") from exc


@dataclass(frozen=True)
class FeatureRecord:
    feature: np.ndarray
    detector_covariance: np.ndarray
    loop_covariance: np.ndarray
    covariance_norm: float
    maximum_covariance_eigenvalue: float
    mean_photon_number: float
    backend: str = "piquasso.GaussianSimulator"


class GaussianLoopReservoir:
    """Persistent N-mode loop coupled to a fresh N-mode ancilla pulse.

    Separate fixed passive and two-mode-active networks act on retained and
    detector arms. Features always come from the detector arm. The unconditional
    loop reduction is retained before measurement. Finite-shot homodyne uses a
    second identical execution, avoiding conditional loop back-action.
    """

    def __init__(self, config: CVConfig) -> None:
        self.config = config
        self._pq = _piquasso()
        rng = np.random.default_rng(config.seed)
        self.loop_unitary = self._haar(config.modes, rng)
        self.detector_unitary = self._haar(config.modes, rng)
        count = max(1, config.modes - 1)
        self.loop_pair_phases = rng.uniform(0, 2 * np.pi, count)
        self.detector_pair_phases = rng.uniform(0, 2 * np.pi, count)
        self._mean: np.ndarray | None = None
        self._covariance: np.ndarray | None = None
        self.execution_count = 0

    @staticmethod
    def _haar(d: int, rng: np.random.Generator) -> np.ndarray:
        z = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
        q, r = np.linalg.qr(z)
        diagonal = np.diag(r)
        return q * (diagonal / np.where(np.abs(diagonal) == 0, 1, np.abs(diagonal)))

    @property
    def parameters(self) -> dict[str, object]:
        return {
            "loop_unitary_real": self.loop_unitary.real.tolist(),
            "loop_unitary_imag": self.loop_unitary.imag.tolist(),
            "detector_unitary_real": self.detector_unitary.real.tolist(),
            "detector_unitary_imag": self.detector_unitary.imag.tolist(),
            "loop_pair_phases": self.loop_pair_phases.tolist(),
            "detector_pair_phases": self.detector_pair_phases.tolist(),
        }

    def reset(self, covariance_scale: float = 1.0) -> None:
        """Initialize a thermal loop; scale one is vacuum."""
        if covariance_scale < 1:
            raise ValueError("covariance_scale must be at least one")
        n = self.config.modes
        self._mean = np.zeros(2 * n)
        self._covariance = 2 * covariance_scale * np.eye(2 * n)

    def _input_gate(self, value: float):
        pq = self._pq
        if self.config.encoding == "angle":
            return pq.Squeezing(r=self.config.input_squeezing, phi=3 * np.pi * value / 4)
        if self.config.encoding == "amplitude":
            amplitude = max(0.0, self.config.input_squeezing * (value + 1) / 2)
            return pq.Squeezing(r=amplitude)
        return pq.Displacement(r=self.config.input_squeezing * value, phi=0.0)

    def _program(self, value: float, mean: np.ndarray, covariance: np.ndarray, measure: bool):
        pq, n = self._pq, self.config.modes
        joint_mean = np.concatenate([mean, np.zeros(2 * n)])
        # Piquasso's default hbar is 2. Preparation instructions multiply mean
        # by sqrt(hbar) and covariance by hbar, so convert stored physical
        # moments back to their dimensionless instruction parameters here.
        joint_covariance = 2 * np.eye(4 * n)
        joint_covariance[: 2 * n, : 2 * n] = covariance
        theta = np.arccos(np.sqrt(self.config.loop_reflectivity))
        with pq.Program() as program:
            pq.Q() | pq.Vacuum()
            pq.Q() | pq.Mean(joint_mean / np.sqrt(2))
            pq.Q() | pq.Covariance(joint_covariance / 2)
            for mode in range(n):
                pq.Q(n + mode) | self._input_gate(value)
                pq.Q(mode, n + mode) | pq.Beamsplitter(theta=theta)
            pq.Q(*range(n)) | pq.Interferometer(self.loop_unitary)
            pq.Q(*range(n, 2 * n)) | pq.Interferometer(self.detector_unitary)
            for offset in range(n - 1):
                pq.Q(offset, offset + 1) | pq.Squeezing2(
                    r=self.config.active_squeezing, phi=self.loop_pair_phases[offset]
                )
                pq.Q(n + offset, n + offset + 1) | pq.Squeezing2(
                    r=self.config.active_squeezing,
                    phi=self.detector_pair_phases[offset],
                )
            if self.config.local_squeezing:
                for mode in range(2 * n):
                    pq.Q(mode) | pq.Squeezing(r=self.config.local_squeezing)
            if self.config.loss:
                angle = np.arccos(np.sqrt(1 - self.config.loss))
                for mode in range(2 * n):
                    pq.Q(mode) | pq.Attenuator(theta=angle)
            if measure:
                pq.Q(*range(n, 2 * n)) | pq.HomodyneMeasurement(phi=0.0)
        return program

    def step(self, value: float) -> FeatureRecord:
        """Execute one physical timestep and extract detector-arm covariance."""
        if self._mean is None or self._covariance is None:
            self.reset()
        assert self._mean is not None and self._covariance is not None
        old_mean, old_covariance = self._mean.copy(), self._covariance.copy()
        pq, n = self._pq, self.config.modes
        simulator = pq.GaussianSimulator(
            d=2 * n,
            config=pq.Config(seed_sequence=self.config.seed + self.execution_count),
        )
        result = simulator.execute(self._program(value, old_mean, old_covariance, False))
        self.execution_count += 1
        loop = result.state.reduced(tuple(range(n)))
        detector = result.state.reduced(tuple(range(n, 2 * n)))
        self._mean = loop.xpxp_mean_vector.copy()
        self._covariance = loop.xpxp_covariance_matrix.copy()
        detector_x = detector.xpxp_covariance_matrix[0::2, 0::2]
        if self.config.measurement == "finite_shot":
            measured = simulator.execute(
                self._program(value, old_mean, old_covariance, True),
                shots=self.config.shots,
            )
            self.execution_count += 1
            samples = np.asarray(measured.samples, dtype=float)[:, 0::2]
            detector_x = np.atleast_2d(np.cov(samples, rowvar=False, ddof=1))
        indices = np.triu_indices(n)
        maximum = float(np.linalg.eigvalsh(self._covariance).max())
        return FeatureRecord(
            feature=detector_x[indices],
            detector_covariance=detector_x,
            loop_covariance=self._covariance.copy(),
            covariance_norm=float(np.linalg.norm(self._covariance)),
            maximum_covariance_eigenvalue=maximum,
            mean_photon_number=float(loop.mean_photon_number()),
        )

    def transform(
        self, values: np.ndarray, washout: int = 0
    ) -> tuple[np.ndarray, list[FeatureRecord]]:
        """Execute a causal input sequence and apply washout."""
        self.reset()
        records = [self.step(float(value)) for value in values]
        if any(
            r.maximum_covariance_eigenvalue > self.config.stability_covariance_limit
            for r in records
        ):
            raise RuntimeError("unstable reservoir exceeded covariance limit")
        return np.asarray([r.feature for r in records[washout:]]), records
