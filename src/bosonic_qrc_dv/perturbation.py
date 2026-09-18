"""Physically unitary hardware perturbations."""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm


def perturb_unitary(unitary: np.ndarray, strength: float, seed: int) -> np.ndarray:
    """Return exp(strength*A)U for a seeded anti-Hermitian generator A."""
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=unitary.shape) + 1j * rng.normal(size=unitary.shape)
    generator = matrix - matrix.conj().T
    generator /= max(np.linalg.norm(generator), np.finfo(float).eps)
    perturbed = expm(strength * generator) @ unitary
    if not np.allclose(perturbed.conj().T @ perturbed, np.eye(len(unitary)), atol=1e-10):
        raise RuntimeError("unitary perturbation invariant failed")
    return perturbed
