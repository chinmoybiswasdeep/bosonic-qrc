"""Validated DV reservoir configuration."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


@dataclass(frozen=True)
class DVConfig:
    modes: int = 4
    photons: int = 2
    seed: int = 13
    measurement: Literal["exact", "finite_shot"] = "exact"
    shots: int = 2000
    include_lower_sectors: bool = True

    def __post_init__(self) -> None:
        if self.modes < 2:
            raise ValueError("modes must be at least two")
        if not 1 <= self.photons <= self.modes:
            raise ValueError("photons must be between one and modes")
        if self.measurement == "finite_shot" and self.shots < 1:
            raise ValueError("shots must be positive")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
