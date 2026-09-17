"""Explicit input windows and chronological split ownership."""

from collections import deque

import numpy as np

from .base import MeasurementBasedReservoir, input_vector
from .config import integer


def chronological_splits(length, *, gap=0, fractions=(0.6, 0.2), washout=0):
    integer(length, "length")
    integer(gap, "gap", 0)
    integer(washout, "washout", 0)
    if len(fractions) != 2 or min(fractions) <= 0 or sum(fractions) >= 1:
        raise ValueError("Train/validation fractions must be positive and sum to <1")
    a, b = int(length * fractions[0]), int(length * sum(fractions))
    bounds = ((0, a), (a + gap, b), (b + gap, length))
    if any(stop - start <= washout for start, stop in bounds):
        raise ValueError("Empty split after gap/washout")
    return {
        name: np.arange(start, stop)
        for name, (start, stop) in zip(("train", "validation", "test"), bounds, strict=True)
    }


def delay_features(inputs, window):
    integer(window, "window")
    u = np.asarray(inputs, float)
    if u.ndim == 1:
        u = u[:, None]
    if u.ndim != 2 or not np.isfinite(u).all():
        raise ValueError("Inputs must be finite with shape (time,channels)")
    out = np.zeros((len(u), window * u.shape[1]))
    for lag in range(window):
        if lag < len(u):
            out[lag:, lag * u.shape[1] : (lag + 1) * u.shape[1]] = u[: len(u) - lag]
    return out


class WindowedMBQELM(MeasurementBasedReservoir):
    """Fresh MBQC resource per explicit trailing window; no persistent quantum memory."""

    def __init__(self, reservoir_factory, window=4):
        integer(window, "window")
        self.factory, self.window = reservoir_factory, window
        self.model = reservoir_factory()
        self.config = self.model.config
        self.reset()

    def reset(self, *, seed=None):
        self.seed = self.config.seed if seed is None else seed
        integer(self.seed, "seed", 0)
        self.history: deque = deque(maxlen=self.window)
        self.model.reset(seed=self.seed)
        return self

    def feature_names(self):
        return self.model.feature_names()

    def summary(self):
        return {**super().summary(), "temporal_model": "explicit-window", "window": self.window}

    def step(self, input_value, *, shots=None):
        value = input_vector(input_value, self.config.input_channels)
        self.history.append(value.copy())
        self.model.reset(seed=self.seed)
        result = None
        for item in self.history:
            result = self.model.step(item, shots=shots)
        assert result is not None
        result.diagnostics["temporal_model"] = "explicit-window; fresh quantum resource"
        result.resources["window"] = len(self.history)
        result.resources["fresh_nodes"] *= len(self.history)
        return result
