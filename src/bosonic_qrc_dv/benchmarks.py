"""Deterministic static boundary benchmarks for DV QRP."""
from __future__ import annotations

import numpy as np


def xor_dataset(repeats: int, seed: int, noise: float = 0.08):
    rng = np.random.default_rng(seed)
    centers = np.asarray([[-1, -1], [-1, 1], [1, -1], [1, 1]], dtype=float)
    labels = np.asarray([0, 1, 1, 0])
    points = np.repeat(centers, repeats, axis=0)
    points += rng.normal(0, noise, points.shape)
    return points, np.repeat(labels, repeats)


def concentric_rings(samples: int, seed: int, noise: float = 0.05):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, samples)
    radii = 0.55 + 0.5 * labels + rng.normal(0, noise, samples)
    angles = rng.uniform(0, 2 * np.pi, samples)
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)]), labels


def intertwined_spirals(samples: int, seed: int, noise: float = 0.04):
    rng = np.random.default_rng(seed)
    labels = np.arange(samples) % 2
    radius = rng.uniform(0.1, 1.0, samples)
    angle = 3.5 * np.pi * radius + labels * np.pi
    points = np.column_stack([radius * np.cos(angle), radius * np.sin(angle)])
    points += rng.normal(0, noise, points.shape)
    return points, labels


def stratified_split(labels: np.ndarray, seed: int, fractions=(0.6, 0.2, 0.2)):
    if not np.isclose(sum(fractions), 1):
        raise ValueError("split fractions must sum to one")
    rng = np.random.default_rng(seed)
    splits = [[], [], []]
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        first = int(fractions[0] * len(indices))
        second = first + int(fractions[1] * len(indices))
        for destination, values in zip(splits, np.split(indices, [first, second])):
            destination.extend(values.tolist())
    return tuple(np.asarray(sorted(values)) for values in splits)


def boundary_fragmentation(prediction_grid: np.ndarray) -> int:
    """Count horizontal/vertical label transitions on a fixed grid."""
    if prediction_grid.ndim != 2:
        raise ValueError("prediction_grid must be two-dimensional")
    horizontal = np.count_nonzero(prediction_grid[:, 1:] != prediction_grid[:, :-1])
    vertical = np.count_nonzero(prediction_grid[1:, :] != prediction_grid[:-1, :])
    return int(horizontal + vertical)

