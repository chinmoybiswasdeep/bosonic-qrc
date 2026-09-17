"""Train-only standardization and an unpenalized ridge intercept."""

import numpy as np


class RidgeReadout:
    def __init__(self, regularization=1e-4):
        if not np.isfinite(regularization) or regularization <= 0:
            raise ValueError("Ridge regularization must be finite and positive")
        self.regularization = regularization
        self.weights = None

    @staticmethod
    def design(inputs, features):
        u, x = np.asarray(inputs, float), np.asarray(features, float)
        if u.ndim == 1:
            u = u[:, None]
        if u.ndim != 2 or x.ndim != 2 or len(u) != len(x):
            raise ValueError("Inputs/features must have matching sample axes")
        result = np.column_stack([u, x])
        if not len(result) or not np.isfinite(result).all():
            raise ValueError("Design must be nonempty and finite")
        return result

    def fit(self, inputs, features, targets):
        x = self.design(inputs, features)
        y = np.asarray(targets, float)
        if y.ndim not in (1, 2) or len(y) != len(x) or not np.isfinite(y).all():
            raise ValueError("Targets must be finite with matching sample axis")
        self.mean = x.mean(axis=0)
        self.scale = x.std(axis=0)
        self.scale[self.scale < 1e-12] = 1
        z = (x - self.mean) / self.scale
        self.target_mean = y.mean(axis=0)
        # SVD avoids squaring condition numbers and handles constant columns.
        u, s, vh = np.linalg.svd(z, full_matrices=False)
        multiplier = s / (s * s + self.regularization)
        self.weights = (vh.T * multiplier) @ (u.T @ (y - self.target_mean))
        return self

    def predict(self, inputs, features):
        if self.weights is None:
            raise ValueError("Fit readout on training data first")
        x = self.design(inputs, features)
        if x.shape[1] != len(self.mean):
            raise ValueError("Feature dimension differs from training")
        return (x - self.mean) / self.scale @ self.weights + self.target_mean
