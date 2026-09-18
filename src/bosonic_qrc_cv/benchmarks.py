"""Deterministic benchmark generators and classical control feature maps."""
from __future__ import annotations

from itertools import combinations_with_replacement

import numpy as np


def narma10(length: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Standard NARMA10 with u~Uniform[0,0.5]."""
    rng = np.random.default_rng(seed)
    u = rng.uniform(0.0, 0.5, length + 10)
    y = np.zeros(length + 10)
    for t in range(9, length + 9):
        y[t + 1] = (
            0.3 * y[t]
            + 0.05 * y[t] * np.sum(y[t - 9 : t + 1])
            + 1.5 * u[t - 9] * u[t]
            + 0.1
        )
    return u[10:], y[10:]


def nonlinear_channel(
    length: int, seed: int, snr_db: float = 20.0
) -> tuple[np.ndarray, np.ndarray]:
    """Canonical four-symbol nonlinear channel-equalization benchmark."""
    rng = np.random.default_rng(seed)
    symbols = rng.choice(np.asarray([-3.0, -1.0, 1.0, 3.0]), length + 10)
    q = np.zeros_like(symbols)
    coefficients = {2: 0.08, 1: -0.12, 0: 1.0, -1: 0.18, -2: -0.10}
    for t in range(2, length + 7):
        q[t] = sum(weight * symbols[t + offset] for offset, weight in coefficients.items())
    observed = q + 0.036 * q**2 - 0.011 * q**3
    signal_power = np.mean(observed[2 : length + 7] ** 2)
    noise_std = np.sqrt(signal_power / 10 ** (snr_db / 10))
    observed += rng.normal(0, noise_std, len(observed))
    return observed[2 : length + 2], symbols[2 : length + 2]


def mackey_glass(
    length: int,
    seed: int,
    tau: int = 17,
    beta: float = 0.2,
    gamma: float = 0.1,
    exponent: int = 10,
    dt: float = 1.0,
) -> np.ndarray:
    """Euler-discretized Mackey-Glass series after a seeded warm-up."""
    rng = np.random.default_rng(seed)
    values = np.full(length + tau + 100, 1.2)
    values[: tau + 1] += rng.normal(0, 0.01, tau + 1)
    for t in range(tau, len(values) - 1):
        delayed = values[t - tau]
        derivative = beta * delayed / (1 + delayed**exponent) - gamma * values[t]
        values[t + 1] = values[t] + dt * derivative
    return values[tau + 100 :]


def tapped_delay_features(inputs: np.ndarray, taps: int) -> np.ndarray:
    if taps < 1:
        raise ValueError("taps must be positive")
    padded = np.pad(inputs, (taps - 1, 0))
    return np.column_stack(
        [padded[taps - 1 - delay : len(padded) - delay] for delay in range(taps)]
    )


def random_nonlinear_features(
    inputs: np.ndarray, dimension: int, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    weights = rng.normal(size=(1, dimension))
    bias = rng.uniform(-np.pi, np.pi, dimension)
    return np.tanh(inputs[:, None] @ weights + bias)


def esn_features(
    inputs: np.ndarray,
    dimension: int,
    seed: int,
    spectral_radius: float = 0.9,
    leak: float = 0.5,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    recurrent = rng.normal(size=(dimension, dimension))
    radius = np.max(np.abs(np.linalg.eigvals(recurrent)))
    recurrent *= spectral_radius / radius
    input_weights = rng.normal(size=dimension)
    state = np.zeros(dimension)
    rows = []
    for value in inputs:
        proposal = np.tanh(recurrent @ state + input_weights * value)
        state = (1 - leak) * state + leak * proposal
        rows.append(state.copy())
    return np.asarray(rows)


def nvar_features(inputs: np.ndarray, taps: int, degree: int = 2) -> np.ndarray:
    delayed = tapped_delay_features(inputs, taps)
    columns = [np.ones(len(inputs))]
    for order in range(1, degree + 1):
        for indices in combinations_with_replacement(range(taps), order):
            columns.append(np.prod(delayed[:, indices], axis=1))
    return np.column_stack(columns)


def classical_gaussian_features(
    inputs: np.ndarray, dimension: int, seed: int
) -> np.ndarray:
    """Matched-size classical Gaussian random-feature control."""
    rng = np.random.default_rng(seed)
    linear = rng.normal(size=dimension)
    quadratic = rng.normal(size=dimension)
    bias = rng.normal(size=dimension)
    return (
        inputs[:, None] * linear
        + inputs[:, None] ** 2 * quadratic
        + bias
    )


def regression_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = target - prediction
    variance = np.var(target)
    nmse = float(np.mean(error**2) / variance)
    return {
        "nmse": nmse,
        "nrmse": float(np.sqrt(nmse)),
        "r2": float(1 - np.sum(error**2) / np.sum((target - target.mean()) ** 2)),
    }

