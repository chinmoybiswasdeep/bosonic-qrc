"""Backend-independent information-processing-capacity mathematics."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import ceil

import numpy as np
from numpy.polynomial.legendre import legval


@dataclass(frozen=True)
class TargetDefinition:
    multi_index: tuple[int, ...]
    degree: int
    maximum_delay: int
    interaction_order: int


@dataclass(frozen=True)
class RankDiagnostics:
    singular_values: np.ndarray
    numerical_rank: int
    absolute_tolerance: float
    relative_tolerance: float
    threshold: float
    condition_number: float
    stable_rank: float
    participation_rank: float


@dataclass(frozen=True)
class LinearFit:
    coefficients: np.ndarray
    feature_mean: np.ndarray
    target_mean: float
    diagnostics: RankDiagnostics

    @property
    def singular_values(self) -> np.ndarray:
        return self.diagnostics.singular_values

    @property
    def numerical_rank(self) -> int:
        return self.diagnostics.numerical_rank

    @property
    def condition_number(self) -> float:
        return self.diagnostics.condition_number

    @property
    def tolerance(self) -> float:
        return self.diagnostics.threshold

    def predict(self, features: np.ndarray) -> np.ndarray:
        return (features - self.feature_mean) @ self.coefficients + self.target_mean


def normalized_legendre(degree: int, values: np.ndarray) -> np.ndarray:
    """Evaluate sqrt(2n+1) P_n, orthonormal for Uniform[-1, 1]."""
    coefficients = np.zeros(degree + 1)
    coefficients[-1] = 1
    return np.sqrt(2 * degree + 1) * legval(values, coefficients)


def enumerate_targets(
    maximum_degree: int,
    maximum_delay: int,
    maximum_interaction_order: int,
    include_cross_delay: bool = True,
) -> list[TargetDefinition]:
    """Enumerate a complete basis; biased partial truncation is unsupported."""
    targets: list[TargetDefinition] = []
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
    return sorted(
        targets,
        key=lambda item: (
            item.degree,
            item.maximum_delay,
            item.interaction_order,
            item.multi_index,
        ),
    )


def evaluate_target(inputs: np.ndarray, target: TargetDefinition) -> np.ndarray:
    delay = len(target.multi_index) - 1
    values = np.ones(len(inputs) - delay)
    for tau, degree in enumerate(target.multi_index):
        if degree:
            values *= normalized_legendre(
                degree, inputs[delay - tau : len(inputs) - tau]
            )
    return values


def squared_correlation(target: np.ndarray, prediction: np.ndarray) -> float:
    centered_target = target - np.mean(target)
    centered_prediction = prediction - np.mean(prediction)
    denominator = np.dot(centered_target, centered_target) * np.dot(
        centered_prediction, centered_prediction
    )
    if denominator <= np.finfo(float).eps:
        return 0.0
    return float(np.dot(centered_target, centered_prediction) ** 2 / denominator)


def test_r2(target: np.ndarray, prediction: np.ndarray) -> float:
    denominator = np.sum((target - np.mean(target)) ** 2)
    if denominator <= np.finfo(float).eps:
        return 0.0
    return float(1 - np.sum((target - prediction) ** 2) / denominator)


def conventional_capacity(target: np.ndarray, prediction: np.ndarray) -> float:
    """Conventional held-out capacity C=max(0, R^2)."""
    return max(0.0, test_r2(target, prediction))


def rank_diagnostics(
    features: np.ndarray,
    relative_tolerance: float = 1e-10,
    absolute_tolerance: float = 1e-12,
) -> RankDiagnostics:
    centered = features - features.mean(axis=0)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    leading = float(singular_values[0]) if singular_values.size else 0.0
    threshold = max(absolute_tolerance, relative_tolerance * leading)
    retained = singular_values[singular_values > threshold]
    rank = int(retained.size)
    condition = float(retained[0] / retained[-1]) if rank else float("inf")
    squared = singular_values**2
    stable = float(squared.sum() / squared.max()) if np.any(squared) else 0.0
    participation = (
        float(squared.sum() ** 2 / np.sum(squared**2)) if np.any(squared) else 0.0
    )
    return RankDiagnostics(
        singular_values,
        rank,
        absolute_tolerance,
        relative_tolerance,
        threshold,
        condition,
        stable,
        participation,
    )


def fit_pseudoinverse(
    features: np.ndarray,
    target: np.ndarray,
    relative_tolerance: float = 1e-10,
    absolute_tolerance: float = 1e-12,
) -> LinearFit:
    feature_mean = features.mean(axis=0)
    target_mean = float(target.mean())
    centered = features - feature_mean
    u, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    diagnostics = rank_diagnostics(features, relative_tolerance, absolute_tolerance)
    retained = singular_values > diagnostics.threshold
    inverse = np.zeros_like(singular_values)
    inverse[retained] = 1 / singular_values[retained]
    coefficients = vh.T @ (inverse * (u.T @ (target - target_mean)))
    return LinearFit(coefficients, feature_mean, target_mean, diagnostics)


def select_ridge_alpha(
    train_features: np.ndarray,
    train_target: np.ndarray,
    validation_features: np.ndarray,
    validation_target: np.ndarray,
    alphas: tuple[float, ...],
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Select ridge alpha on validation data using train-only scaling."""
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
        coefficients = np.linalg.pinv(train.T @ train + alpha * identity) @ (
            train.T @ centered_target
        )
        prediction = validation @ coefficients + target_mean
        loss = float(np.mean((validation_target - prediction) ** 2))
        if best is None or loss < best[0]:
            best = (loss, alpha, coefficients)
    if best is None:
        raise ValueError("ridge alpha grid must not be empty")
    return best[1], best[2], mean, scale


def benjamini_hochberg(p_values: np.ndarray, q: float = 0.05) -> np.ndarray:
    order = np.argsort(p_values)
    sorted_p = p_values[order]
    passing = sorted_p <= q * np.arange(1, len(p_values) + 1) / len(p_values)
    significant = np.zeros(len(p_values), dtype=bool)
    if np.any(passing):
        cutoff = np.flatnonzero(passing)[-1]
        significant[order[: cutoff + 1]] = True
    return significant


def minimum_null_surrogates(number_of_targets: int, q: float) -> int:
    """Minimum B satisfying 1/(B+1) <= q/M."""
    return max(1, ceil(number_of_targets / q - 1))


def independent_null_capacities(
    train_features: np.ndarray,
    validation_features: np.ndarray,
    test_features: np.ndarray,
    target: TargetDefinition,
    count: int,
    seed: int,
    relative_tolerance: float,
    ridge_alphas: tuple[float, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Regenerate independent IID input targets and rerun both readouts."""
    rng = np.random.default_rng(seed)
    delay = len(target.multi_index) - 1
    pinv_values = np.empty(count)
    ridge_values = np.empty(count)
    for index in range(count):
        train = evaluate_target(rng.uniform(-1, 1, len(train_features) + delay), target)
        validation = evaluate_target(
            rng.uniform(-1, 1, len(validation_features) + delay), target
        )
        held = evaluate_target(rng.uniform(-1, 1, len(test_features) + delay), target)
        fit = fit_pseudoinverse(train_features, train, relative_tolerance)
        pinv_values[index] = conventional_capacity(held, fit.predict(test_features))
        _, coefficients, mean, scale = select_ridge_alpha(
            train_features,
            train,
            validation_features,
            validation,
            ridge_alphas,
        )
        prediction = (test_features - mean) / scale @ coefficients + train.mean()
        ridge_values[index] = conventional_capacity(held, prediction)
    return pinv_values, ridge_values


def target_bank_diagnostics(targets: np.ndarray) -> dict[str, object]:
    means = targets.mean(axis=1)
    variances = targets.var(axis=1)
    scale = np.sqrt(np.where(variances > 0, variances, 1.0))
    standardized = (targets - means[:, None]) / scale[:, None]
    gram = standardized @ standardized.T / targets.shape[1]
    eigenvalues = np.linalg.eigvalsh(gram)
    positive = eigenvalues[eigenvalues > 1e-12]
    condition = float(positive[-1] / positive[0]) if positive.size else float("inf")
    effective = (
        float(eigenvalues.sum() ** 2 / np.sum(eigenvalues**2))
        if np.any(eigenvalues)
        else 0.0
    )
    off_diagonal = gram - np.diag(np.diag(gram))
    return {
        "means": means.tolist(),
        "variances": variances.tolist(),
        "gram_matrix": gram.tolist(),
        "gram_eigenvalues": eigenvalues.tolist(),
        "gram_condition_number": condition,
        "effective_independent_target_count": effective,
        "maximum_absolute_off_diagonal": float(np.max(np.abs(off_diagonal))),
    }


def max_statistic_threshold(null_matrix: np.ndarray, alpha: float = 0.05) -> float:
    if null_matrix.ndim != 2:
        raise ValueError("null_matrix must have shape (targets, surrogates)")
    return float(np.quantile(np.max(null_matrix, axis=0), 1 - alpha))


def hierarchical_bootstrap(
    values: np.ndarray,
    reservoir_ids: np.ndarray,
    data_ids: np.ndarray,
    seed: int = 0,
    replicates: int = 2000,
) -> dict[str, object]:
    """Bootstrap reservoir seeds, then data seeds within each reservoir."""
    values = np.asarray(values, dtype=float)
    reservoir_ids = np.asarray(reservoir_ids)
    data_ids = np.asarray(data_ids)
    reservoirs = np.unique(reservoir_ids)
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates)
    for index in range(replicates):
        sampled_reservoirs = rng.choice(reservoirs, len(reservoirs), replace=True)
        selected: list[float] = []
        for reservoir in sampled_reservoirs:
            mask = reservoir_ids == reservoir
            available_data = np.unique(data_ids[mask])
            sampled_data = rng.choice(available_data, len(available_data), replace=True)
            for data in sampled_data:
                candidates = values[mask & (data_ids == data)]
                selected.append(float(rng.choice(candidates)))
        draws[index] = np.mean(selected)
    q1, q3 = np.quantile(values, [0.25, 0.75])
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "standard_deviation": (
            float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        ),
        "iqr": float(q3 - q1),
        "bootstrap_95_ci": np.quantile(draws, [0.025, 0.975]).tolist(),
        "bootstrap_replicates": replicates,
    }


def capacity_bound(
    total_capacity: float, numerical_rank: int, tolerance: float = 1e-6
) -> tuple[bool, float]:
    residual = float(numerical_rank - total_capacity)
    return residual >= -tolerance, residual
