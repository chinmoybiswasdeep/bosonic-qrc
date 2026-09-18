"""Multimode interacting measurement-based Fock reservoir."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import comb

import numpy as np
import photographiq as pg
from photographiq.backends.mixed_fock import MixedFockBackend

from .core import (
    bose_hubbard_parts,
    exact_unitary,
    floquet_unitary,
    state_diagnostics,
    strang_unitary,
)


@dataclass(frozen=True)
class MBFockConfig:
    memory_modes: int = 2
    cutoff: int = 5
    hopping: float = 1.0
    interaction: float = 0.8
    disorder: tuple[float, ...] = (0.0, 0.0)
    interval: float = 0.2
    boundary: str = "open"
    evolution_model: str = "exact"
    trotter_substeps: int = 4
    coupling: float = 0.12
    input_scale: float = 0.25
    cat_amplitude: float = 0.3
    feedforward: float = 0.08
    transmissivity: float = 0.94
    evolution: str = "unconditional"
    resource_ablation: str = "cat"
    cubic_strength: float = 0.03
    seed: int = 17
    max_matrix_bytes: int = 128_000_000

    def __post_init__(self):
        if self.memory_modes < 1 or self.cutoff < 2:
            raise ValueError("memory_modes positive and cutoff>=2 required")
        if len(self.disorder) != self.memory_modes:
            raise ValueError("one onsite energy per memory mode required")
        if self.evolution_model not in {"exact", "floquet", "trotter"}:
            raise ValueError("unknown evolution model")
        if self.evolution not in {"unconditional", "conditional"}:
            raise ValueError("unknown trajectory policy")
        if self.resource_ablation not in {"cat", "cat_cubic", "number"}:
            raise ValueError("unsupported resource ablation")
        if not 0 <= self.transmissivity <= 1:
            raise ValueError("invalid transmissivity")
        dimension = comb(self.cutoff + self.memory_modes - 1, self.memory_modes)
        if 16 * dimension**2 > self.max_matrix_bytes:
            raise MemoryError("memory density matrix exceeds configured budget")


class InteractingMBReservoir:
    """PhotoGraphiQ collision/measurement plus exact interacting memory layer."""

    def __init__(self, config: MBFockConfig | None = None):
        self.config = config or MBFockConfig()
        self.nodes = tuple(range(self.config.memory_modes))
        self.engine_class = MixedFockBackend
        self.reset()

    def reset(self, seed: int | None = None):
        self.rng = np.random.default_rng(self.config.seed if seed is None else seed)
        vacuum = ((0,) * self.config.memory_modes,)
        self.state = pg.FockDensityMatrix([[1.0]], vacuum)
        self.time = 0
        return self

    def _pattern(self, value: float):
        ancilla = self.config.memory_modes
        if self.config.resource_ablation == "number":
            resource = pg.FockInput.number(1)
        else:
            resource = pg.FockInput.cat(self.config.cat_amplitude, self.config.cutoff)
        pattern = pg.Pattern(inputs=self.nodes)
        pattern.append(pg.Prepare(ancilla, state=resource))
        pattern.append(pg.Displace(ancilla, q=2 * self.config.input_scale * value))
        if self.config.resource_ablation == "cat_cubic":
            pattern.append(pg.CubicPhase(ancilla, self.config.cubic_strength))
        for node in self.nodes:
            pattern.append(
                pg.BeamSplitter(
                    node,
                    ancilla,
                    self.config.coupling / np.sqrt(self.config.memory_modes),
                )
            )
        pattern.append(pg.Measure(ancilla, pg.PhotonNumber(), "count"))
        for node in self.nodes:
            pattern.append(pg.Rotate(node, self.config.feedforward * pg.Outcome("count")))
        pattern.append(pg.Output(self.nodes))
        return pattern

    def _simulate_branch(self, value: float, outcome: int | None):
        engine = self.engine_class(
            self.config.cutoff, max_matrix_bytes=self.config.max_matrix_bytes
        )
        return pg.simulate(
            self._pattern(value),
            initial_state=self.state,
            backend=engine,
            seed=int(self.rng.integers(0, 2**63)),
            measurement_outcomes=None if outcome is None else {"count": outcome},
        )

    def _collision(self, value: float, outcome: int | None):
        if self.config.evolution == "conditional":
            result = self._simulate_branch(value, outcome)
            probability = float(result.measurement_statistics["count"]["value"])
            return result.state, result.outcomes["count"], [probability], probability
        weighted = None
        basis = None
        probabilities = []
        for count in range(self.config.cutoff):
            try:
                result = self._simulate_branch(value, count)
            except ValueError as error:
                if "zero" in str(error).lower():
                    probabilities.append(0.0)
                    continue
                raise
            probability = float(result.measurement_statistics["count"]["value"])
            probabilities.append(probability)
            if basis is None:
                basis = result.state.basis
                weighted = np.zeros_like(result.state.density_matrix)
            if result.state.basis != basis:
                raise RuntimeError("PNR branches returned incompatible bases")
            weighted += probability * result.state.density_matrix
        retained = float(sum(probabilities))
        if retained <= 0:
            raise RuntimeError("all PNR branches have zero probability")
        assert weighted is not None and basis is not None
        return pg.FockDensityMatrix(weighted / retained, basis), None, probabilities, retained

    def _interact(self, state):
        basis = state.basis
        density_input = np.asarray(
            state.matrix if isinstance(state, pg.FockDensityMatrix) else state.density_matrix
        )
        parts = bose_hubbard_parts(
            basis,
            self.config.hopping,
            self.config.interaction,
            np.asarray(self.config.disorder),
            self.config.boundary,
        )
        if self.config.evolution_model == "exact":
            unitary = exact_unitary(parts.total, self.config.interval)
        elif self.config.evolution_model == "floquet":
            unitary = floquet_unitary(parts, self.config.interval)
        else:
            unitary = strang_unitary(parts, self.config.interval, self.config.trotter_substeps)
        density = unitary @ density_input @ unitary.conj().T
        return pg.FockDensityMatrix(density, basis)

    def _loss(self, state):
        pattern = pg.Pattern(inputs=self.nodes)
        for node in self.nodes:
            pattern.append(pg.Loss(node, self.config.transmissivity))
        pattern.append(pg.Output(self.nodes))
        return pg.simulate(
            pattern,
            initial_state=state,
            backend=self.engine_class(
                self.config.cutoff, max_matrix_bytes=self.config.max_matrix_bytes
            ),
            seed=int(self.rng.integers(0, 2**63)),
        ).state

    def step(self, value: float, outcome: int | None = None):
        collision, selected, probabilities, retained = self._collision(float(value), outcome)
        output = self._loss(self._interact(collision))
        self.state = pg.FockDensityMatrix(output.density_matrix, output.basis)
        self.time += 1
        rho = np.asarray(self.state.matrix)
        probabilities_diagonal = np.real(np.diag(rho))
        occupations = np.asarray(self.state.basis)
        means = probabilities_diagonal @ occupations
        second = probabilities_diagonal @ occupations**2
        parity = [
            float(probabilities_diagonal @ ((-1.0) ** occupations[:, node])) for node in self.nodes
        ]
        features = np.r_[
            means,
            second,
            parity,
            probabilities_diagonal,
        ]
        return {
            "features": features,
            "outcome": selected,
            "outcome_probabilities": probabilities,
            "diagnostics": {
                **state_diagnostics(rho, self.state.basis, self.config.cutoff),
                "collision_retained_probability": retained,
                "minimum_retained_norm": min(output.retained_norms, default=1.0),
            },
        }

    def provenance(self):
        return {
            "config": asdict(self.config),
            "backend": "PhotoGraphiQ MixedFockBackend for resource, gates, PNR and loss",
            "interacting_memory": "local exact SciPy Bose-Hubbard adapter",
            "cutoff_semantics": "exclusive total photon number",
            "hilbert_dimension": len(self.state.basis),
            "unconditional_policy": "Born-probability weighted unnormalized PNR branches",
        }
