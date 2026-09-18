"""Validated CV experiment configuration."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


@dataclass(frozen=True)
class CVConfig:
    """Fixed Gaussian-loop parameters.

    ``loop_reflectivity`` is the intensity fraction of the old loop field in
    the retained beam-splitter output, so Piquasso uses
    ``theta = arccos(sqrt(loop_reflectivity))``.
    """

    modes: int = 3
    loop_reflectivity: float = 0.45
    input_squeezing: float = 1.0
    active_squeezing: float = 0.05
    local_squeezing: float = 0.0
    loss: float = 0.02
    encoding: Literal["angle", "amplitude", "displacement"] = "angle"
    measurement: Literal["exact", "finite_shot"] = "exact"
    shots: int = 1000
    seed: int = 7
    stability_covariance_limit: float = 1e6

    def __post_init__(self) -> None:
        if self.modes < 1:
            raise ValueError("modes must be positive")
        if not 0 <= self.loop_reflectivity <= 1:
            raise ValueError("loop_reflectivity must be in [0, 1]")
        if not 0 <= self.loss < 1:
            raise ValueError("loss must be in [0, 1)")
        if self.measurement == "finite_shot" and self.shots < 2:
            raise ValueError("finite-shot covariance needs at least two shots")

    @property
    def feature_dimension(self) -> int:
        return self.modes * (self.modes + 1) // 2

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
