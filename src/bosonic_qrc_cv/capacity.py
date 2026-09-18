"""Shared information-processing-capacity mathematics.

This module contains no quantum simulation. It validates targets and classical
linear readouts applied to backend-generated features.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
from numpy.polynomial.legendre import legval


@dataclass(frozen=True)
class TargetDefinition:
    multi_index: tuple[int, ...]
    degree: int
    maximum_delay: int
    interaction_order: int


@dataclass(frozen=True)
class LinearFit:
    coefficients: np.ndarray
    feature_mean: np.ndarray
    target_mean: float
    singular_values: np.ndarray
    numerical_rank: int
    condition_number: float
    tolerance: float

    def predict(self, features: np.ndarray) -> np.ndarray:
        return (features - self.feature_mean) @ self.coefficients + self.target_mean


def normalized_legendre(degree: int, values: np.ndarray) -> np.ndarray:
    """Evaluate sqrt(2n+1) P_n, orthonormal for uniform [-1, 1]."""
    coefficients = np.zeros(degree + 1)
    coefficients[-1] = 1
    return np.sqrt(2 * degree + 1) * legval(values, coefficients)


def enumerate_targets(
    maximum_degree: int,
    maximum_delay: int,
    maximum_interaction_order: int,
    include_cross_delay: bool = True,
    maximum_targets: int | None = None,
) -> list[TargetDefinition]:
    """Deterministically enumerate nonconstant finite Legendre multi-indices."""
    targets = []
    for alpha in product(range(maximum_degree + 1), repeat=maximum_delay + 1):
        degree = sum(alpha)
        active = [delay for delay, order in enumerate(alpha) if order]
        if degree == 0 or degree > maximum_degree:
            continue
        if len(active) > maximum_interaction_order:
            continue
        if not include_cross_delay and len(active) > 1:
            continue
        targets.append(TargetDefinition(alpha, degree, max(active), len(active)))
    targets.sort(key=lambda item: (item.degree, item.maximum_delay, item.interaction_order, item.multi_index))
    return targets if maximum_targets is None else targets[:maximum_targets]


def evaluate_target(inputs: np.ndarray, target: TargetDefinition) -> np.ndarray:
    """Evaluate a causal target; output index zero corresponds to t=max_delay."""
    delay = len(target.multi_index) - 1
    values = np.ones(len(inputs) - delay)
    for tau, degree in enumerate(target.multi_index):
        if degree:
            values *= normalized_legendre(degree, inputs[delay - tau : len(inputs) - tau])
    return values


def squared_correlation(target: np.ndarray, prediction: np.ndarray) -> float:
    centered_target = target - np.mean(target)
    centered_prediction = prediction - np.mean(prediction)
    denominator = np.dot(centered_target, centered_target) * np.dot(centered_prediction, centered_prediction)
    if denominator <= np.finfo(float).eps:
        return 0.0
    return float(np.dot(centered_target, centered_prediction) ** 2 / denominator)


def test_r2(target: np.ndarray, prediction: np.ndarray) -> float:
    denominator = np.sum((target - np.mean(target)) ** 2)
    return float(1 - np.sum((target - prediction) ** 2) / denominator)


def fit_pseudoinverse(features: np.ndarray, target: np.ndarray, relative_tolerance: float = 1e-10) -> LinearFit:
    """Fit a centered SVD readout with an explicit relative tolerance."""
    feature_mean = features.mean(axis=0)
    target_mean = float(target.mean())
    centered = features - feature_mean
    u, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    threshold = (
        relative_tolerance * max(float(singular_values[0]), 1.0)
        if singular_values.size
        else 0
    )
    retained = singular_values > threshold
    inverse = np.zeros_like(singular_values)
    inverse[retained] = 1 / singular_values[retained]
    coefficients = vh.T @ (inverse * (u.T @ (target - target_mean)))
    rank = int(retained.sum())
    condition = float(singular_values[0] / singular_values[rank - 1]) if rank else float("inf")
    return LinearFit(coefficients, feature_mean, target_mean, singular_values, rank, condition, threshold)


def select_ridge_alpha(
    train_features: np.ndarray,
    train_target: np.ndarray,
    validation_features: np.ndarray,
    validation_target: np.ndarray,
    alphas: tuple[float, ...],
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Select ridge alpha only on validation data with train-only scaling."""
    mean = train_features.mean(axis=0)
    scale = train_features.std(axis=0)
    scale[scale == 0] = 1
    train = (train_features - mean) / scale
    validation = (validation_features - mean) / scale
    target_mean = train_target.mean()
    centered_target = train_target - target_mean
    best: tuple[float, float, np.ndarray] | None = None
    identity = np.eye(train.shape[1])
    for alpha in alphas:
        coefficients = np.linalg.pinv(train.T @ train + alpha * identity) @ train.T @ centered_target
        prediction = validation @ coefficients + target_mean
        loss = float(np.mean((validation_target - prediction) ** 2))
        if best is None or loss < best[0]:
            best = (loss, alpha, coefficients)
    assert best is not None
    return best[1], best[2], mean, scale


def benjamini_hochberg(p_values: np.ndarray, q: float = 0.05) -> np.ndarray:
    """Return hypotheses passing Benjamini-Hochberg FDR correction."""
    order = np.argsort(p_values)
    sorted_p = p_values[order]
    passing = sorted_p <= q * np.arange(1, len(p_values) + 1) / len(p_values)
    significant = np.zeros(len(p_values), dtype=bool)
    if np.any(passing):
        cutoff = np.flatnonzero(passing)[-1]
        significant[order[: cutoff + 1]] = True
    return significant


def null_capacities(
    train_features: np.ndarray,
    test_features: np.ndarray,
    train_target: np.ndarray,
    test_target: np.ndarray,
    count: int,
    seed: int,
    relative_tolerance: float,
) -> np.ndarray:
    """Fit circularly shifted targets and return held-out null capacities."""
    rng = np.random.default_rng(seed)
    capacities = np.empty(count)
    for index in range(count):
        train_shift = int(rng.integers(1, len(train_target)))
        test_shift = int(rng.integers(1, len(test_target)))
        shifted_train = np.roll(train_target, train_shift)
        shifted_test = np.roll(test_target, test_shift)
        fit = fit_pseudoinverse(train_features, shifted_train, relative_tolerance)
        capacities[index] = squared_correlation(shifted_test, fit.predict(test_features))
    return capacities


def capacity_bound(total_significant_capacity: float, numerical_rank: int, tolerance: float = 1e-6) -> tuple[bool, float]:
    residual = float(numerical_rank - total_significant_capacity)
    return residual >= -tolerance, residual
