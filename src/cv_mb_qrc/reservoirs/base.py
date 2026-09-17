"""Causal sequence interface; run_sequence continues state unless reset is requested."""

from abc import ABC, abstractmethod
from dataclasses import asdict
from time import perf_counter
from typing import Any

import numpy as np

from .config import integer
from .results import ReservoirResult, environment


def input_vector(value, channels):
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        array = array.reshape(1)
    if array.shape != (channels,) or not np.isfinite(array).all():
        raise ValueError(f"Input must be finite with shape ({channels},)")
    return array


class MeasurementBasedReservoir(ABC):
    config: Any

    @abstractmethod
    def reset(self, *, seed: int | None = None): ...

    @abstractmethod
    def step(self, input_value, *, shots=None) -> ReservoirResult: ...

    @abstractmethod
    def feature_names(self) -> tuple[str, ...]: ...

    def summary(self) -> dict:
        return {
            "model": type(self).__name__,
            "configuration": asdict(self.config),
            "feature_dimension": len(self.feature_names()),
            "temporal_model": "stateful",
        }

    def run_sequence(self, inputs, *, washout=0, shots=None) -> ReservoirResult:
        integer(washout, "washout", 0)
        array = np.asarray(inputs, dtype=float)
        if array.ndim == 1 and self.config.input_channels == 1:
            array = array[:, None]
        if (
            array.ndim != 2
            or array.shape[1] != self.config.input_channels
            or len(array) <= washout
            or not np.isfinite(array).all()
        ):
            raise ValueError("Use nonempty finite (time,channels) inputs and washout < length")
        start = perf_counter()
        rows = [self.step(row, shots=shots) for row in array]
        last = rows[-1]
        return ReservoirResult(
            np.stack([r.features for r in rows[washout:]]),
            self.feature_names(),
            last.evolution,
            last.estimator,
            shots,
            [r.outcomes for r in rows[washout:]],
            {"washout": washout, "step_diagnostics": [r.diagnostics for r in rows[washout:]]},
            {
                **last.resources,
                "temporal_depth": len(array),
                "fresh_nodes_total": sum(r.resources.get("fresh_nodes", 0) for r in rows),
            },
            asdict(self.config),
            environment(),
            perf_counter() - start,
        )
