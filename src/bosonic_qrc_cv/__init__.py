"""Piquasso-backed continuous-variable quantum reservoir computing."""

from .config import CVConfig
from .reservoir import GaussianLoopReservoir

__all__ = ["CVConfig", "GaussianLoopReservoir"]
