"""Temporal tasks and leakage-safe readout protocol for EOC studies."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations_with_replacement

import numpy as np


@dataclass(frozen=True)
class SplitPolicy:
    train: int = 120
    validation: int = 60
    test: int = 120
    washout: int = 20
    embargo: int = 10


def narma10(length: int, seed: int):
    rng = np.random.default_rng(seed)
    inputs = rng.uniform(0, 0.5, length + 10)
    target = np.zeros(length + 10)
    for time in range(9, length + 9):
        target[time + 1] = (
            0.3 * target[time]
            + 0.05 * target[time] * np.sum(target[time - 9 : time + 1])
            + 1.5 * inputs[time - 9] * inputs[time]
            + 0.1
        )
    return inputs[10:], target[10:]


def delayed_parity(length: int, seed: int, delay: int = 3):
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, length + delay)
    target = np.asarray(
        [np.sum(bits[index : index + delay + 1]) % 2 for index in range(length)]
    )
    return bits[delay:].astype(float), target.astype(float)


def nonlinear_channel(length: int, seed: int, snr_db: float = 20):
    rng = np.random.default_rng(seed)
    symbols = rng.choice(np.asarray([-3.0, -1.0, 1.0, 3.0]), length + 8)
    linear = np.zeros_like(symbols)
    weights = {2: 0.08, 1: -0.12, 0: 1.0, -1: 0.18, -2: -0.10}
    for time in range(2, length + 5):
        linear[time] = sum(
            coefficient * symbols[time + offset]
            for offset, coefficient in weights.items()
        )
    observed = linear + 0.036 * linear**2 - 0.011 * linear**3
    power = np.mean(observed[2 : length + 2] ** 2)
    observed += rng.normal(0, np.sqrt(power / 10 ** (snr_db / 10)), len(observed))
    return observed[2 : length + 2], symbols[2 : length + 2]


def mackey_glass(length: int, seed: int, tau: int = 17):
    rng = np.random.default_rng(seed)
    values = np.full(length + tau + 101, 1.2)
    values[: tau + 1] += rng.normal(0, 0.01, tau + 1)
    for time in range(tau, len(values) - 1):
        delayed = values[time - tau]
        values[time + 1] = values[time] + (
            0.2 * delayed / (1 + delayed**10) - 0.1 * values[time]
        )
    series = values[tau + 100 :]
    return series[:-1], series[1:]


def chronological_indices(length: int, policy: SplitPolicy):
    needed = policy.train + policy.validation + policy.test + 2 * policy.embargo
    if length < needed:
        raise ValueError("sequence is too short for split and embargo")
    train = np.arange(0, policy.train)
    validation_start = policy.train + policy.embargo
    validation = np.arange(validation_start, validation_start + policy.validation)
    test_start = validation[-1] + 1 + policy.embargo
    test = np.arange(test_start, test_start + policy.test)
    return train, validation, test


def reservoir_features(factory, inputs: np.ndarray, washout: int):
    model = factory()
    model.reset()
    rows = [model.step(float(value))["features"] for value in inputs]
    return np.asarray(rows)[washout:]


def tapped_delay_features(inputs: np.ndarray, taps: int):
    padded = np.pad(inputs, (taps - 1, 0))
    return np.column_stack(
        [padded[taps - 1 - delay : len(padded) - delay] for delay in range(taps)]
    )


def rff_features(inputs: np.ndarray, dimension: int, seed: int):
    rng = np.random.default_rng(seed)
    return np.sqrt(2 / dimension) * np.cos(
        inputs[:, None] * rng.normal(size=dimension)
        + rng.uniform(0, 2 * np.pi, dimension)
    )


def esn_features(inputs: np.ndarray, dimension: int, seed: int):
    rng = np.random.default_rng(seed)
    recurrent = rng.normal(size=(dimension, dimension))
    recurrent *= 0.9 / np.max(np.abs(np.linalg.eigvals(recurrent)))
    weights = rng.normal(size=dimension)
    state = np.zeros(dimension)
    rows = []
    for value in inputs:
        state = 0.5 * state + 0.5 * np.tanh(recurrent @ state + weights * value)
        rows.append(state.copy())
    return np.asarray(rows)


def nvar_features(inputs: np.ndarray, taps: int, degree: int = 2):
    delayed = tapped_delay_features(inputs, taps)
    columns = [np.ones(len(inputs))]
    for order in range(1, degree + 1):
        for indices in combinations_with_replacement(range(taps), order):
            columns.append(np.prod(delayed[:, indices], axis=1))
    return np.column_stack(columns)


def ipc_targets(inputs: np.ndarray, maximum_delay: int, maximum_degree: int):
    """Return labelled Legendre delay and cross-delay IPC targets."""
    values = np.asarray(inputs, float)
    span = np.ptp(values)
    scaled = (
        np.zeros_like(values) if span == 0 else 2 * (values - values.min()) / span - 1
    )
    delayed = tapped_delay_features(scaled, maximum_delay + 1)
    targets = {}
    for delay in range(1, maximum_delay + 1):
        for degree in range(1, maximum_degree + 1):
            coefficient = np.zeros(degree + 1)
            coefficient[-1] = 1
            label = f"degree={degree};delays={delay}"
            targets[label] = np.polynomial.legendre.legval(
                delayed[:, delay], coefficient
            )
    for left in range(1, maximum_delay):
        for right in range(left + 1, maximum_delay + 1):
            targets[f"degree=2;delays={left},{right}"] = (
                delayed[:, left] * delayed[:, right]
            )
    return targets


def effective_feature_rank(
    features: np.ndarray, relative_threshold: float = 1e-10
) -> int:
    centered = np.asarray(features, float) - np.mean(features, axis=0)
    singular = np.linalg.svd(centered, compute_uv=False)
    if not singular.size or singular[0] == 0:
        return 0
    return int(np.count_nonzero(singular > relative_threshold * singular[0]))


def fit_evaluate(
    features: np.ndarray,
    target: np.ndarray,
    policy: SplitPolicy,
    alphas=(1e-8, 1e-6, 1e-4, 1e-2, 1.0),
):
    train, validation, test = chronological_indices(len(target), policy)
    mean = features[train].mean(axis=0)
    scale = features[train].std(axis=0)
    keep = scale > 1e-12
    scale = scale[keep]
    transformed = (features[:, keep] - mean[keep]) / scale
    target_mean = target[train].mean()
    identity = np.eye(keep.sum())
    candidates = []
    for alpha in alphas:
        coefficients = np.linalg.solve(
            transformed[train].T @ transformed[train] + alpha * identity,
            transformed[train].T @ (target[train] - target_mean),
        )
        prediction = transformed[validation] @ coefficients + target_mean
        candidates.append((np.mean((target[validation] - prediction) ** 2), alpha))
    selected = min(candidates)[1]
    coefficients = np.linalg.solve(
        transformed[train].T @ transformed[train] + selected * identity,
        transformed[train].T @ (target[train] - target_mean),
    )
    prediction = transformed[test] @ coefficients + target_mean
    variance = np.var(target[test])
    nmse = float(np.mean((target[test] - prediction) ** 2) / variance)
    return {
        "selected_alpha": selected,
        "test_nmse": nmse,
        "test_nrmse": float(np.sqrt(nmse)),
        "test_r2": float(1 - nmse),
        "train_indices": train.tolist(),
        "validation_indices": validation.tolist(),
        "test_indices": test.tolist(),
        "preprocessing_fit_indices": train.tolist(),
    }
