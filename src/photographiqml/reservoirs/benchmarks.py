"""Locally generated tasks and causally separated ridge evaluation."""

import numpy as np

from .config import integer
from .readout import RidgeReadout
from .temporal import delay_features


def capacity_targets(inputs, delays=5):
    integer(delays, "delays")
    u = np.asarray(inputs, float)
    history = delay_features(u, delays + 1)
    columns, names = [], []
    for lag in range(1, delays + 1):
        x = history[:, lag]
        columns.extend([np.sqrt(3) * x, np.sqrt(5) * (3 * x * x - 1) / 2])
        names.extend([f"linear_{lag}", f"quadratic_{lag}"])
    for a in range(1, delays + 1):
        for b in range(a + 1, delays + 1):
            columns.append(3 * history[:, a] * history[:, b])
            names.append(f"cross_{a}_{b}")
    return np.column_stack(columns), tuple(names)


def narma10(inputs):
    u = np.asarray(inputs, float)
    if u.ndim != 1 or not np.isfinite(u).all() or np.any((u < 0) | (u > 0.5)):
        raise ValueError("NARMA10 inputs must lie in [0,0.5]")
    y = np.zeros(len(u) + 1)
    for t in range(9, len(u)):
        y[t + 1] = 0.3 * y[t] + 0.05 * y[t] * sum(y[t - 9 : t + 1]) + 1.5 * u[t - 9] * u[t] + 0.1
    if not np.isfinite(y).all():
        raise FloatingPointError("NARMA trajectory diverged")
    return y[1:]


def mackey_glass(length, *, delay=17, dt=0.1, burnin=1000):
    """Euler: dx/dt=.2*x(t-17)/(1+x(t-17)^10)-.1*x; initial history 1.2."""
    integer(length, "length")
    if not np.isfinite([delay, dt]).all() or delay <= 0 or dt <= 0:
        raise ValueError("Positive delay and dt required")
    lag = int(round(delay / dt))
    stride = int(round(1 / dt))
    if lag < 1 or stride < 1:
        raise ValueError("dt must resolve delay and unit sampling interval")
    x = np.full(lag + (length + burnin) * stride + 1, 1.2)
    for t in range(lag, len(x) - 1):
        delayed = x[t - lag]
        x[t + 1] = x[t] + dt * (0.2 * delayed / (1 + delayed**10) - 0.1 * x[t])
    return x[lag + burnin * stride : lag + (burnin + length) * stride : stride]


def metrics(target, prediction):
    y, p = np.asarray(target, float), np.asarray(prediction, float)
    if y.shape != p.shape or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("Finite predictions/targets of identical shape required")
    variance = float(np.mean((y - y.mean()) ** 2))
    if variance <= 0:
        raise ValueError("R2 undefined for constant target")
    nmse = float(np.mean((y - p) ** 2) / variance)
    return {"r2": 1 - nmse, "nmse": nmse, "nrmse": float(np.sqrt(nmse))}


def select_readout(train, validation, regularizations=(1e-6, 1e-4, 1e-2, 1.0)):
    """Only train/validation are accepted; test labels cannot enter selection."""
    candidates = []
    for alpha in regularizations:
        model = RidgeReadout(alpha).fit(*train)
        prediction = model.predict(*validation[:2])
        score = float(np.mean((prediction - validation[2]) ** 2))
        candidates.append((score, model))
    return min(candidates, key=lambda item: item[0])[1]


class ClassicalFeatures:
    """Seeded ESN/RFF/input-only/delay controls with explicit equal histories."""

    def __init__(self, kind, dimension=14, seed=0, window=4):
        if kind not in ("esn", "rff", "input_only", "delay", "persistence"):
            raise ValueError("Unknown baseline")
        integer(dimension, "dimension")
        integer(window, "window")
        self.kind, self.dimension, self.window = kind, dimension, window
        rng = np.random.default_rng(seed)
        self.mask = rng.normal(size=(dimension, window))
        self.bias = rng.uniform(-np.pi, np.pi, dimension)
        self.recurrent = rng.normal(size=(dimension, dimension)) / np.sqrt(dimension)
        self.recurrent *= 0.8 / max(np.linalg.norm(self.recurrent, 2), 1e-12)

    def transform(self, inputs):
        u = np.asarray(inputs, float)
        windows = delay_features(u, self.window)
        if self.kind == "delay":
            return windows
        if self.kind == "persistence":
            return u[:, None]
        if self.kind == "rff":
            return np.sqrt(2 / self.dimension) * np.cos(windows @ self.mask.T + self.bias)
        if self.kind == "input_only":
            return np.tanh(u[:, None] * self.mask[:, 0] + self.bias)
        state, rows = np.zeros(self.dimension), []
        for value in u:
            state = np.tanh(self.recurrent @ state + self.mask[:, 0] * value + self.bias)
            rows.append(state.copy())
        return np.array(rows)
