"""Validated experiment configuration."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


@dataclass(frozen=True)
class CVConfig:
    r"""Parameters for a fixed Gaussian feedback-loop reservoir.

    The input mode is squeezed with ``r=input_scale`` and phase
    :math:`3\pi s_t/4` for ``angle`` encoding.  All values are fixed before
    readout fitting.
    """

    modes: int = 3
    reflectivity: float = 0.45
    input_scale: float = 0.7
    active_squeezing: float = 0.12
    loss: float = 0.0
    encoding: Literal["angle", "amplitude", "displacement"] = "angle"
    seed: int = 7

    def __post_init__(self) -> None:
        if self.modes < 1:
            raise ValueError("modes must be positive")
        if not 0.0 <= self.reflectivity <= 1.0:
            raise ValueError("reflectivity must be in [0, 1]")
        if not 0.0 <= self.loss < 1.0:
            raise ValueError("loss must be in [0, 1)")

    @property
    def feature_dimension(self) -> int:
        return self.modes * (self.modes + 1) // 2

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
